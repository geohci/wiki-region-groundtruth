import argparse
import bz2
from collections import namedtuple
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

PageLinks = namedtuple('PageLinks', ['qid', 'links'])
INLINKS_HEADER = ['qid_from', 'inlinks',]
OUTLINKS_HEADER = ['qid_from', 'outlinks']

class SpatialModel:
    '''
    Description:

    This class holds all the model functions

    '''

    def __init__(self, groundtruth_fn, inlink_file, outlink_file, data_dir,
                 train_ratio, val_ratio, test_ratio, skip_splits,
                 threshold, iteration, updated_model_fn):
        '''
        Description:
        This contains the default values passed into the model

        Input parameters:
            - inlink_file: This is inlink file to be passed into model
            - outlink_file: This is outlink_file to be passed into model
            - test_ratio: This is ratio of test set to be used in train-test split
            - threshold: This is the threshold used in training model
            - iteration: This is the number of times model will be trained before predictions are done on test size

        '''
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "Train/Val/Test split must add to 1"
        self.threshold = threshold
        self.iteration = iteration
        self.updates_per_iter = []

        if inlink_file and outlink_file:
            logging.info("Using both inlinks and outlinks.")
        elif inlink_file:
            logging.info("Using inlinks only.")
        elif outlink_file:
            logging.info("Using outlinks only.")
        else:
            raise Exception("Must provide inlinks and/or outlinks.")

        if not skip_splits:
            self.train_test_split(inlink_file, outlink_file)
        logging.info("Loading groundtruth")
        self.load_groundtruth(groundtruth_fn)
        logging.info(f"{len(self.groundtruth)} QIDs in groundtruth.")
        logging.info("Calculating idf for regions. Initial biases:")
        self.get_region_occurrences()
        for c in sorted(self.country_idf, key=self.country_idf.get, reverse=True):
            logging.info(f'{self.integer_country_dict[c]}: {self.country_idf[c]}')

        split_type = 'test'
        if self.val_ratio:
            split_type = 'val'

        logging.info(f"\n== Baseline evaluation on {split_type} set ({time.ctime()}) ==")
        self.evaluate_model(split_type, full_table=False)  # establish baseline
        for i in range(1, self.iteration+1):
            logging.info(f"\n== Iteration {i}/{self.iteration} ({time.ctime()}) ==")
            self.train_model()
            logging.info(f"Evaluation on {split_type} set after iteration {i}/{self.iteration}")
            self.evaluate_model(split_type, full_table=False)
            self.get_region_occurrences()  # update idf calculations to account for expanded groundtruth
            logging.info("Updated IDF -- top 5:")
            for c in sorted(self.country_idf, key=self.country_idf.get, reverse=True)[:5]:
                logging.info(f'{self.integer_country_dict[c]}: {self.country_idf[c]}')
            if self.updates_per_iter[-1][2] / self.updates_per_iter[-1][0] < 0.01:
                logging.info("Stopping early. Updated less than 1% of QIDs on most recent pass.")
                break

        if self.test_ratio:
            logging.info("\n== Final evaluation on test set ==")
            self.evaluate_model('test', full_table=True)

        logging.info("\nProcessing statistics by iteration #:")
        update_df = pd.DataFrame(self.updates_per_iter, columns=['Processed', 'Coverage', 'Updated', 'Added to GT'],
                                 index=[i for i in range(1, len(self.updates_per_iter) + 1)])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            logging.info(update_df)

        if updated_model_fn:
            with bz2.open(os.path.join(data_dir, updated_model_fn), 'wt') as fout:
                tsvwriter = csv.writer(fout, delimiter='\t')
                tsvwriter.writerow(['item', 'countries'])
                for qid in sorted(self.groundtruth, key=lambda x: int(x[1:])):
                    tsvwriter.writerow([qid,
                                        "|".join([self.integer_country_dict[ridx] for ridx in self.groundtruth[qid]])])

    def load_groundtruth(self, groundtruth_fn=None):
        '''
        Description:
        This function takes in compressed groundtruth and returns a cleaned dictionary data.

        Input parameters:
            - groundtruth: This contains the groundtruth file

        Output:
            Returns a dictionary containing keys as QID/Article Page ID and values as regions

        '''
        self.groundtruth = {}

        with bz2.open(groundtruth_fn, "rt") as file:
            tsvreader = csv.reader(file, delimiter='\t')
            header = next(tsvreader)
            assert 'item' in header and 'countries' in header, \
                f"Groundtruth must at least have `item` and `countries` columns: {header}"
            qid_idx = header.index('item')
            countries_idx = header.index('countries')
            self.country_integer_dict = {}
            self.integer_country_dict = {}

            for line in tsvreader:
                qid = line[qid_idx]
                regions = line[countries_idx].split('|')
                region_indices = []
                if regions:
                    for r in regions:
                        if r not in self.country_integer_dict:
                            idx = len(self.country_integer_dict)
                            self.country_integer_dict[r] = idx
                            self.integer_country_dict[idx] = r
                        region_indices.append(self.country_integer_dict[r])
                    self.groundtruth[qid] = tuple(region_indices)

    def generate_statistics(self, country, initial_result, final_result):
        '''
        Description:

        Takes in both result lists and outputs a tuple containing TP,FP,FN,
        Recall,F1 score and Precision

        Input parameters:
            - country: Contains initial country dict
            - initial_result: Result generated from initial groundtruth
            - final_result: Result generated from final groundtruth


        Output:
            - Country dict containing keys as region and values as metrics for model evaluation calculation.


        '''

        for var in final_result:
            if country.get(var, None):
                continue
            else:
                country[var] = {'tp': 0, 'fp': 0, 'fn': 0}

            if var in initial_result:
                country[var]['tp'] += 1
            else:
                country[var]['fp'] += 1

        for var_sec in initial_result:
            if country.get(var_sec, None):
                continue
            else:
                country[var_sec] = {'tp': 0, 'fp': 0, 'fn': 0}

            if var_sec not in final_result:
                country[var_sec]['fn'] += 1

        return country

    def get_region_occurrences(self):
        """
        Determine how often each region appears in the initial groundtruth to correct for bias.
        :return:
        """
        self.country_idf = {}
        for qid in self.groundtruth:
            for region_idx in self.groundtruth[qid]:
                self.country_idf[region_idx] = self.country_idf.get(region_idx, 0) + 1

        num_pages = len(self.groundtruth)
        # idf
        #self.country_idf = {c: np.log(num_pages / self.country_idf[c]) for c in self.country_idf}

        # simple baseline -- % of articles that have each country label (subtract this out)
        self.country_idf = {c: self.country_idf[c] / num_pages for c in self.country_idf}

    def inlinks_reader(self, inlinks_fn=None):
        with bz2.open(inlinks_fn, 'rt') as ifin:
            i_reader = csv.reader(ifin, delimiter='\t')
            assert next(i_reader) == INLINKS_HEADER
            for line in i_reader:
                pl = PageLinks(*line)
                if pl.qid:
                    yield(pl)

    def outlinks_reader(self, outlinks_fn=None):
        with bz2.open(outlinks_fn, 'rt') as ofin:
            o_reader = csv.reader(ofin, delimiter='\t')
            assert next(o_reader) == OUTLINKS_HEADER
            for line in o_reader:
                pl = PageLinks(*line)
                if pl.qid:
                    yield(pl)

    def align_inlinks_outlinks(self, inlinks_fn, outlinks_fn):
        '''
        Description:
        This generator is used to align the inlinks and outlinks file based on page IDs/QIDs

        Input:
            - inlinks_reader: bzipped inlink TSV filename
            - outlinks_reader: bzipped outlink TSV filename

        Yields:
            Tuple containing inlink and outlink lines (as list or None)
        '''
        if os.path.exists(inlinks_fn):
            i_reader = self.inlinks_reader(inlinks_fn)
            inlinks = next(i_reader)
        else:
            inlinks = None
        if os.path.exists(outlinks_fn):
            o_reader = self.outlinks_reader(outlinks_fn)
            outlinks = next(o_reader)
        else:
            outlinks = None

#        processed = 0
        while True:
#            if processed == 1000:
#                break
#            processed += 1
            # No more pages -- stop iteration
            if not inlinks and not outlinks:
                break
            # no more inlinks; continue just w/ outlinks
            elif not inlinks:
                yield (None, outlinks)
                try:
                    outlinks = next(o_reader)
                except StopIteration:
                    outlinks = None
            # no more outlinks; continue just w/ inlinks
            elif not outlinks:
                yield (inlinks, None)
                try:
                    inlinks = next(i_reader)
                except StopIteration:
                    inlinks = None
            # both inlinks and outlinks; align
            else:
                iqid = inlinks.qid  # inlinks QID
                oqid = outlinks.qid  # outlinks QID
                # same page
                if iqid == oqid:
                    yield (inlinks, outlinks)
                # different pages; outlinks missing so yield just inlinks
                elif iqid < oqid:
                    yield(inlinks, None)
                    try:
                        inlinks = next(i_reader)
                    except StopIteration:
                        inlinks = None
                # different pages; inlinks missing so yield just outlinks
                else:
                    yield (None, outlinks)
                    try:
                        outlinks = next(o_reader)
                    except StopIteration:
                        outlinks = None

    def normalize_regions(self, region_counts, num_links):
        normed_r_counts = {}
        for r in region_counts:
            if region_counts[r] > 1:
                # tf-idf
                #normed_r_counts[r] = (region_counts[r] / num_links) * self.country_idf[r]
                # simple baseline
                normed_r_counts[r] = (region_counts[r] / num_links) - self.country_idf[r]
        return normed_r_counts

    def predict_regions(self, region_counts):
        predictions = []
        for r in region_counts:
            if region_counts[r] >= self.threshold:
                predictions.append(r)
        return predictions

    def extract_region_summary(self, links):
        '''
        Description:
            This is a helper function that brings other functions needed to get the region summary

        Input parameters:
            - groundtruth:
            - link: This is the inlink/outlink
            - integer_country_dict: This is the integer to country mapping to decode groundtruth

        Output:
            This returns a dict containing link summary


        '''
        # count up occurrences of each region in links
        split_links = links.split(' ')
        region_counts = {}
        for link in split_links:
            link_regions = self.groundtruth.get(link, [])
            for r in link_regions:
                region_counts[r] = region_counts.get(r, 0) + 1

        # make adjustments
        num_links = len(split_links)
        normed_page_regions = self.normalize_regions(region_counts, num_links)

        return normed_page_regions

    def get_fn(self, split_type='train', link_type='inlinks'):
        if split_type not in ('train', 'val', 'test'):
            raise ValueError("Train type must be one of: train, val, test")
        if link_type not in ('inlinks', 'outlinks'):
            raise ValueError("Link type must be one of: inlinks, outlinks")

        return os.path.join(self.data_dir, f'{split_type}_{link_type}.tsv.bz2')

    def train_test_split(self, inlinks_fn=None, outlinks_fn=None):
        '''
        Description:
            Given a specified test ratio, this splits the inlink and outlink file into training and test files

        Input parameters:
            - inlink_file: Compressed inlink file
            - outlink_file: Compressed outlink file
            - test_ratio: Takes in a test ratio which splits files according to it

        Output:
            - train_inlinks: Inlink training data
            - test_inlinks: Inlink test data
            - train_outlinks: Outlink training data
            - test_outlinks: Outlink test data

        '''
        logging.info(f"Splitting data into: train {self.train_ratio} -- val {self.val_ratio} -- test {self.test_ratio}.")

        # Initialize train/val/test files
        if inlinks_fn:
            if self.train_ratio:
                train_inlink_file = bz2.open(self.get_fn('train', 'inlinks'), 'wt')
                train_inlink_writer = csv.writer(train_inlink_file, delimiter='\t')
                train_inlink_writer.writerow(INLINKS_HEADER)
            if self.val_ratio:
                val_inlink_file = bz2.open(self.get_fn('val', 'inlinks'), 'wt')
                val_inlink_writer = csv.writer(val_inlink_file, delimiter='\t')
                val_inlink_writer.writerow(INLINKS_HEADER)
            if self.test_ratio:
                test_inlink_file = bz2.open(self.get_fn('test', 'inlinks'), 'wt')
                test_inlink_writer = csv.writer(test_inlink_file, delimiter='\t')
                test_inlink_writer.writerow(INLINKS_HEADER)

        if outlinks_fn:
            if self.train_ratio:
                train_outlink_file = bz2.open(self.get_fn('train', 'outlinks'), 'wt')
                train_outlink_writer = csv.writer(train_outlink_file, delimiter='\t')
                train_outlink_writer.writerow(OUTLINKS_HEADER)
            if self.val_ratio:
                val_outlink_file = bz2.open(self.get_fn('val', 'outlinks'), 'wt')
                val_outlink_writer = csv.writer(val_outlink_file, delimiter='\t')
                val_outlink_writer.writerow(OUTLINKS_HEADER)
            if self.test_ratio:
                test_outlink_file = bz2.open(self.get_fn('test', 'outlinks'), 'wt')
                test_outlink_writer = csv.writer(test_outlink_file, delimiter='\t')
                test_outlink_writer.writerow(OUTLINKS_HEADER)

        train_count = 0
        val_count = 0
        test_count = 0

        # Iterate through inlinks and outlinks file
        for i, (inlinks, outlinks) in enumerate(self.align_inlinks_outlinks(inlinks_fn, outlinks_fn), start=1):
            if i % 100000 == 0:
                logging.debug('{0} lines processed'.format(i))

            # Generate random number from 0 and 1
            random_num = random.random()

            # Check if train_ratio is greater than random number and append to test data
            if random_num <= self.train_ratio:
                train_count += 1
                if inlinks:
                    train_inlink_writer.writerow(inlinks)
                if outlinks:
                    train_outlink_writer.writerow(outlinks)
            elif random_num <= self.train_ratio + self.val_ratio:
                val_count += 1
                if inlinks:
                    val_inlink_writer.writerow(inlinks)
                if outlinks:
                    val_outlink_writer.writerow(outlinks)
            else:
                test_count += 1
                if inlinks:
                    test_inlink_writer.writerow(inlinks)
                if outlinks:
                    test_outlink_writer.writerow(outlinks)

        logging.info(f'Train {train_count} ({train_count / i:.3f}), val {val_count} ({val_count / i:.3f}), test {test_count} ({test_count / i:.3f})')

        # Close files
        if inlinks_fn:
            if self.train_ratio:
               train_inlink_file.close()
            if self.val_ratio:
               val_inlink_file.close()
            if self.test_ratio:
               test_inlink_file.close()
        if outlinks_fn:
            if self.train_ratio:
               train_outlink_file.close()
            if self.val_ratio:
                val_outlink_file.close()
            if self.test_ratio:
                test_outlink_file.close()

    def train_model(self):
        '''
        Description:
        This function is primarily used in training the model

        Input parameters:
            - train_inlinks: Inlink training data
            - train_outlinks: Outlink training data


        Output:
            Returns updated groundtruth

        '''
        gt_update = {}
        processed = 0
        for (inlinks, outlinks) in self.align_inlinks_outlinks(self.get_fn('train', 'inlinks'),
                                                               self.get_fn('train', 'outlinks')):
            processed += 1
            if inlinks and outlinks:
                # Get inlink region summary
                i_page_regions = self.extract_region_summary(inlinks.links)
                o_page_regions = self.extract_region_summary(outlinks.links)
                for r in o_page_regions:
                    i_page_regions[r] = i_page_regions.get(r, 0) + o_page_regions[r]
                for r in i_page_regions:
                    i_page_regions[r] = i_page_regions[r] / 2

                region_predictions = self.predict_regions(i_page_regions)
                qid = inlinks.qid

            elif outlinks:
                page_regions = self.extract_region_summary(outlinks.links)
                region_predictions = self.predict_regions(page_regions)
                qid = outlinks.qid
            elif inlinks:
                page_regions = self.extract_region_summary(inlinks.links)
                region_predictions = self.predict_regions(page_regions)
                qid = inlinks.qid
            else:
                raise Exception('No inlinks or outlinks.')

            if region_predictions:
                gt_update[qid] = region_predictions

        updated = 0
        added_to_gt = 0
        for qid in gt_update:
            do_update = False
            for c in gt_update[qid]:
                if c not in self.groundtruth.get(qid, []):
                    do_update = True
                    break
            if do_update:
                updated += 1
                if qid not in self.groundtruth:
                    added_to_gt += 1
                self.groundtruth[qid] = tuple(set(gt_update[qid] + list(self.groundtruth.get(qid, []))))

        self.updates_per_iter.append([processed, len(gt_update), updated, added_to_gt])
        logging.info((f"{processed} processed. "
                      f"{len(gt_update)} ({len(gt_update) / processed:.3f}) with predictions. "
                      f"{updated} ({updated / processed:.3f}) updated. "
                      f"{added_to_gt} ({added_to_gt / processed:.3f}) added to groundtruth"))

    def evaluate_model(self, split_type='val', full_table=None, max_examples=5):
        '''
        Description:
        This evaluates the current state of the groundtruth.

        Input parameters:

        Output:
            Returns the summary stats


        '''
        country_stats = {}
        for country_idx in self.integer_country_dict:
            country_stats[country_idx] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0}

        examples_shown = 0
        for i, (inlinks, outlinks) in enumerate(self.align_inlinks_outlinks(self.get_fn(split_type, 'inlinks'),
                                                                            self.get_fn(split_type, 'outlinks'))):
            if inlinks and outlinks:
                # Get inlink region summary
                page_regions = self.extract_region_summary(inlinks.links)
                o_page_regions = self.extract_region_summary(outlinks.links)
                for r in o_page_regions:
                    page_regions[r] = page_regions.get(r, 0) + o_page_regions[r]
                for r in page_regions:
                    page_regions[r] = page_regions[r] / 2
                region_predictions = self.predict_regions(page_regions)
                qid = inlinks.qid
            elif outlinks:
                page_regions = self.extract_region_summary(outlinks.links)
                region_predictions = self.predict_regions(page_regions)
                qid = outlinks.qid
            elif inlinks:
                page_regions = self.extract_region_summary(inlinks.links)
                region_predictions = self.predict_regions(page_regions)
                qid = inlinks.qid
            else:
                raise Exception("No inlinks or outlinks.")

            gt = self.groundtruth.get(qid, [])
            if examples_shown < max_examples and random.random() < 0.0001:
                logging.debug(f"Example qid={qid}")
                logging.debug(f"gt={[self.integer_country_dict[idx] for idx in gt]}")
                logging.debug(f"pred={[self.integer_country_dict[idx] for idx in region_predictions]}")
                logging.debug(f"regions={[(self.integer_country_dict[idx], '{0:.3f}'.format(page_regions[idx])) for idx in sorted(page_regions, key=page_regions.get, reverse=True)]}")
                examples_shown += 1

            for c in country_stats:
                if c in gt and c in region_predictions:
                    country_stats[c]['n'] += 1
                    country_stats[c]['TP'] += 1
                elif c in gt:
                    country_stats[c]['n'] += 1
                    country_stats[c]['FN'] += 1
                elif c in region_predictions:
                    country_stats[c]['FP'] += 1
                else:
                    country_stats[c]['TN'] += 1

        for c in country_stats:
            s = country_stats[c]
            try:
                s['precision'] = s['TP'] / (s['TP'] + s['FP'])
            except ZeroDivisionError:
                s['precision'] = 0
            try:
                s['recall'] = s['TP'] / (s['TP'] + s['FN'])
            except ZeroDivisionError:
                s['recall'] = 0
            try:
                s['f1'] = 2 * (s['precision'] * s['recall']) / (s['precision'] + s['recall'])
            except ZeroDivisionError:
                s['f1'] = 0

        stats_df = pd.DataFrame(country_stats).T
        stats_df['country'] = [self.integer_country_dict[c_idx] for c_idx in stats_df.index]
        stats_df.set_index('country', inplace=True)
        if full_table:
            stats_df.sort_values(by='n', inplace=True, ascending=False)
            stats_df[''] = '-->'
            stats_df = stats_df[['n', '', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1']]
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
                logging.info(stats_df)
            #if args.mlc_res_tsv:
            #    stats_df.to_csv(args.mlc_res_tsv, sep='|')

        null_vals = list(stats_df[stats_df.isnull().T.any()].index)
        if null_vals:
            logging.info("Dropping {0} because of Null values.".format(null_vals))
        stats_df = stats_df.dropna()
        logging.info(("\nPrecision: {0:.3f} micro; {1:.3f} macro\n"
                      "Recall: {2:.3f} micro; {3:.3f} macro\n"
                      "F1: {4:.3f} micro; {5:.3f} macro").format(
            np.average(stats_df['precision'], weights=stats_df['n']), np.mean(stats_df['precision']),
            np.average(stats_df['recall'], weights=stats_df['n']), np.mean(stats_df['recall']),
            np.average(stats_df['f1'], weights=stats_df['n']), np.mean(stats_df['f1'])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_fn")
    parser.add_argument("--inlink_file", default="")
    parser.add_argument("--outlink_file", default="")
    parser.add_argument("--data_dir", default='./')
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--skip_splits", action="store_true", default=False, help="Assume train/val/test splits already exist.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--updated_model_fn")
    parser.add_argument("--log_fn")
    args = parser.parse_args()

    # "%(asctime)s %(filename)s: %(levelname)8s %(message)s"
    if args.log_fn:
        logging.basicConfig(filename=args.log_fn, level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.debug(f"Starting: {time.ctime()}")
    logging.debug(args)

    SpatialModel(groundtruth_fn=args.groundtruth_fn,
                 inlink_file=args.inlink_file,
                 outlink_file=args.outlink_file,
                 data_dir=args.data_dir,
                 train_ratio=args.train_ratio,
                 val_ratio=args.val_ratio,
                 test_ratio=args.test_ratio,
                 skip_splits=args.skip_splits,
                 threshold=args.threshold,
                 iteration=args.iteration,
                 updated_model_fn=args.updated_model_fn)