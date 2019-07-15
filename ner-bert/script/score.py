from __future__ import absolute_import, division, print_function
import pyarrow.parquet as pq
import faulthandler
import logging
import pandas as pd
import json
import os
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from .arg_opts import score_opts
from .utils import plot, serialize_result, save_as_df


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
faulthandler.enable()
logging.info(f"Load pyarrow.parquet explicitly: {pq}")
logger = logging.getLogger(__name__)


class Ner:

    def __init__(self, model_dir: str, meta: dict = {}):
        self.no_cuda = True if meta.get('No cuda', 'False') == 'True' else False
        self.local_rank = int(meta.get('Local Rank', -1))
        self.test_batch_size = int(meta.get('Test Batch Size', 8))
        self.output_eval_dir = meta.get('Output evaluation results', '')
        self.model, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]

        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.model.eval()
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        self.model.to(self.device)

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=model_config["num_labels"])
        if torch.cuda.is_available() and not self.no_cuda:
            model.load_state_dict(torch.load(output_model_file))
        else:
            model.load_state_dict(torch.load(output_model_file, map_location='cpu'))

        return model, model_config

    def run(self, test_features: pd.DataFrame, meta: dict = None):
        # Load features
        raw_text_list = test_features['raw_text'].tolist()
        input_ids_list = test_features['input_ids'].tolist()
        input_mask_list = test_features['input_mask'].tolist()
        segment_ids_list = test_features['segment_ids'].tolist()
        valid_positions_list = test_features['valid_positions'].tolist()

        logger.info("***** Running scoring *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", self.test_batch_size)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask_list, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
        all_valid_positions = torch.tensor(valid_positions_list, dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_valid_positions)
        # Run prediction for test data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.test_batch_size)

        y_pred = []
        for input_ids, input_mask, segment_ids, valid_positions in tqdm(test_dataloader, desc="Predicting"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            valid_positions = valid_positions.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            input_mask = input_mask.to('cpu').numpy()
            valid_positions = valid_positions.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_2 = []
                for j, m in enumerate(mask):
                    if (j == 0) or (j == len(mask)-1):
                        continue
                    if m:
                        if valid_positions[i][j] != -1:
                            temp_2.append(self.label_map[logits[i][j]])
                    else:
                        temp_2.pop()
                        break
                y_pred.append(temp_2)
        assert len(y_pred) == len(raw_text_list)
        df_pred = serialize_result(raw_text_list, y_pred)
        print(df_pred)
        return df_pred

    def evaluation(self, test_features: pd.DataFrame):
        # Load features
        raw_text_list = test_features['raw_text'].tolist()
        input_ids_list = test_features['input_ids'].tolist()
        input_mask_list = test_features['input_mask'].tolist()
        segment_ids_list = test_features['segment_ids'].tolist()
        label_ids_list = test_features['label_ids'].tolist()

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", self.test_batch_size)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask_list, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
        all_label_ids = torch.tensor(label_ids_list, dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for test data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.test_batch_size)

        y_pred = []
        y_true = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            input_mask = input_mask.to('cpu').numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(mask):
                    if (j == 0) or (j == len(mask) - 1):
                        continue
                    if m:
                        if self.label_map[label_ids[i][j]] != "X":
                            temp_1.append(self.label_map[label_ids[i][j]])
                            temp_2.append(self.label_map[logits[i][j]])
                    else:
                        temp_1.pop()
                        temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
        assert len(y_pred) == len(raw_text_list)
        df_pred = serialize_result(raw_text_list, y_pred)
        print(df_pred)
        # Output prediction results
        save_as_df(df_pred, self.output_eval_dir)
        # Plot
        plot(y_true, y_pred, self.output_eval_dir)
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(self.output_eval_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("\n%s", report)
            writer.write(report)


if __name__ == "__main__":
    parser = score_opts()
    args, _ = parser.parse_known_args()

    meta = {'No cuda': str(args.no_cuda),
            'Local Rank': str(args.local_rank),
            'Test Batch Size': str(args.test_batch_size),
            'Output evaluation results': str(args.output_eval_dir)}
    # Load features
    test_features = pd.read_parquet(os.path.join(args.test_feature_dir, "feature.parquet"), engine='pyarrow')
    header_names = set(list(test_features.columns.values))
    ner_task = Ner(model_dir=args.trained_model_dir, meta=meta)
    if "label_ids" not in header_names:
        df_pred = ner_task.run(test_features=test_features)
        # Output prediction results
        save_as_df(df_pred, ner_task.output_eval_dir)
    else:
        ner_task.evaluation(test_features=test_features)

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.output_eval_dir, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


