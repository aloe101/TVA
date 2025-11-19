import json
import numpy as np
import pandas as pd
import multiprocessing as mp

from .builder import EVALUATORS, remove_duplicate_annotations


@EVALUATORS.register_module()
class mAP:
    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        top_k=None,
        blocked_videos=None,
        filt_gt=None,
        thumos=1,
        thread=16,
        dataset=None,
    ):
        super().__init__()

        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")

        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.gt_fields = ["database"]
        self.pred_fields = ["results"]
        self.filt_gt = filt_gt
        self.thread = thread  # multi-process workers

        # Get blocked videos
        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        # Import ground truth and predictions.
        
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)
        if self.filt_gt and dataset == 'charades':
            self.ground_truth = self.filt()
            activity_index = {'Sitting in a chair': 0, 'Someone is going from standing to sitting': 1, 'Sitting at a table': 2, 'Watching a laptop or something on a laptop': 3, 'Closing a box': 4, 'Opening a box': 5, 'Taking a box from somewhere': 6, 'Sitting on sofa/couch': 7, 'Someone is standing up from somewhere': 8, 'Holding a box': 9, 'Putting a box somewhere': 10, 'Throwing a box somewhere': 11, 'Holding a phone/camera': 12, 'Sitting in a bed': 13, 'Taking a blanket from somewhere': 14, 'Throwing a blanket somewhere': 15, 'Taking a pillow from somewhere': 16, 'Putting a blanket somewhere': 17, 'Tidying some clothes': 18, 'Tidying up a blanket/s': 19, 'Holding a towel/s': 20, 'Tidying something on the floor': 21, 'Taking some clothes from somewhere': 22, 'Holding some clothes': 23, 'Someone is dressing': 24, 'Putting clothes somewhere': 25, 'Putting a towel/s somewhere': 26, 'Throwing clothes somewhere': 27, 'Taking a towel/s from somewhere': 28, 'Holding some food': 29, 'Holding a mirror': 30, 'Someone is running somewhere': 31, 'Someone is smiling': 32, 'Holding a dish': 33, 'Holding a book': 34, 'Taking food from somewhere': 35, 'Walking through a doorway': 36, 'Holding a sandwich': 37, 'Playing with a phone/camera': 38, 'Taking a picture of something': 39, 'Holding a cup/glass/bottle of something': 40, 'Drinking from a cup/glass/bottle': 41, 'Watching/looking at a picture': 42, 'Taking off some shoes': 43, 'Putting shoes somewhere': 44, 'Someone is undressing': 45, 'Washing their hands': 46, 'Working/Playing on a laptop': 47, 'Working at a table': 48, 'Putting a cup/glass/bottle somewhere': 49, 'Taking a cup/glass/bottle from somewhere': 50, 'Holding a bag': 51, 'Taking a bag from somewhere': 52, 'Closing a closet/cabinet': 53, 'Opening a closet/cabinet': 54, 'Washing a cup/glass/bottle': 55, 'Opening a door': 56, 'Closing a door': 57, 'Holding a broom': 58, 'Taking a broom from somewhere': 59, 'Fixing a light': 60, 'Holding a laptop': 61, 'Washing something with a towel': 62, 'Taking a laptop from somewhere': 63, 'Opening a laptop': 64, 'Turning off a light': 65, 'Putting something on a table': 66, 'Taking paper/notebook from somewhere': 67, 'Pouring something into a cup/glass/bottle': 68, 'Someone is holding a paper/notebook': 69, 'Taking a dish/es from somewhere': 70, 'Working on paper/notebook': 71, 'Putting their paper/notebook somewhere': 72, 'Tidying up a towel/s': 73, 'Tidying a shelf or something on a shelf': 74, 'Putting some food somewhere': 75, 'Tidying up a closet/cabinet': 76, 'Putting groceries somewhere': 77, 'Putting something on a shelf': 78, 'Someone is eating something': 79, 'Someone is laughing': 80, 'Putting a pillow somewhere': 81, 'Holding a blanket': 82, 'Opening a bag': 83, 'Putting a sandwich somewhere': 84, 'Tidying up with a broom': 85, 'Someone is sneezing': 86, 'Grasping onto a doorknob': 87, 'Closing a laptop': 88, 'Putting a laptop somewhere': 89, 'Eating a sandwich': 90, 'Holding a pillow': 91, 'Watching/Looking outside of a window': 92, 'Talking on a phone/camera': 93, 'Fixing their hair': 94, 'Taking a phone/camera from somewhere': 95, 'Watching something/someone/themselves in a mirror': 96, 'Sitting on a table': 97, 'Turning on a light': 98, 'Putting a dish/es somewhere': 99, 'Snuggling with a pillow': 100, 'Watching television': 101, 'Throwing a pillow somewhere': 102, 'Taking/consuming some medicine': 103, 'Closing a window': 104, 'Sitting on the floor': 105, 'Holding some medicine': 106, 'Putting a broom somewhere': 107, 'Someone is cooking something': 108, 'Making a sandwich': 109, 'Holding a shoe/shoes': 110, 'Throwing shoes somewhere': 111, 'Taking shoes from somewhere': 112, 'Fixing a door': 113, 'Tidying up a table': 114, 'Taking a sandwich from somewhere': 115, 'Smiling at a book': 116, 'Watching/Reading/Looking at a book': 117, 'Opening a book': 118, 'Taking a book from somewhere': 119, 'Someone is awakening somewhere': 120, 'Someone is awakening in bed': 121, 'Lying on a bed': 122, 'Lying on a sofa/couch': 123, 'Snuggling with a blanket': 124, 'Putting a phone/camera somewhere': 125, 'Throwing a book somewhere': 126, 'Wash a dish/dishes': 127, 'Throwing food somewhere': 128, 'Throwing a towel/s somewhere': 129, 'Holding a vacuum': 130, 'Fixing a vacuum': 131, 'Closing a refrigerator': 132, 'Opening a refrigerator': 133, 'Holding a picture': 134, 'Reaching for and grabbing a picture': 135, 'Laughing at a picture': 136, 'Putting a picture somewhere': 137, 'Putting a book somewhere': 138, 'Throwing something on the floor': 139, 'Putting a bag somewhere': 140, 'Washing some clothes': 141, 'Putting on shoe/shoes': 142, 'Taking something from a box': 143, 'Lying on the floor': 144, 'Closing a book': 145, 'Throwing a bag somewhere': 146, 'Laughing at television': 147, 'Fixing a doorknob': 148, 'Smiling in a mirror': 149, 'Standing on a chair': 150, 'Throwing a broom somewhere': 151, 'Opening a window': 152, 'Washing a table': 153, 'Taking a vacuum from somewhere': 154, 'Washing a mirror': 155, 'Washing a window': 156}
        elif self.filt_gt and thumos:
            self.ground_truth = self.filt()
            activity_index = {
    'CricketBowling': 0, 'CricketShot': 1, 'VolleyballSpiking': 2, 'JavelinThrow': 3,
    'Shotput': 4, 'TennisSwing': 5, 'GolfSwing': 6, 'ThrowDiscus': 7, 'Billiards': 8,
    'CleanAndJerk': 9, 'LongJump': 10, 'Diving': 11, 'CliffDiving': 12,
    'BasketballDunk': 13, 'HighJump': 14, 'SoccerPenalty': 15, 'BaseballPitch': 16,
    'HammerThrow': 17, 'FrisbeeCatch': 18, 'PoleVault': 19
}
        elif self.filt_gt and not thumos:
            activity_index = {'Beer pong': 0, 'Kneeling': 1, 'Tumbling': 2, 'Sharpening knives': 3, 'Playing water polo': 4, 'Scuba diving': 5, 'Arm wrestling': 6, 'Playing bagpipes': 7, 'Riding bumper cars': 8, 'Surfing': 9, 'Hopscotch': 10, 'Gargling mouthwash': 11, 'Playing violin': 12, 'Plastering': 13, 'Changing car wheel': 14, 'Horseback riding': 15, 'Playing congas': 16, 'Walking the dog': 17, 'Rafting': 18, 'Hurling': 19, 'Removing curlers': 20, 'Playing beach volleyball': 21, 'Windsurfing': 22, 'Using parallel bars': 23, 'Playing drums': 24, 'Playing badminton': 25, 'Getting a piercing': 26, 'Camel ride': 27, 'Sailing': 28, 'Wrapping presents': 29, 'Hand washing clothes': 30, 'Braiding hair': 31, 'Longboarding': 32, 'Doing motocross': 33, 'Vacuuming floor': 34, 'Blow-drying hair': 35, 'Smoking hookah': 36, 'Doing fencing': 37, 'Playing harmonica': 38, 'Spinning': 39, 'Playing blackjack': 40, 'Discus throw': 41, 'Playing flauta': 42, 'Swimming': 43, 'Ice fishing': 44, 'Spread mulch': 45, 'Canoeing': 46, 'Mowing the lawn': 47, 'Capoeira': 48, 'Preparing salad': 49, 'Beach soccer': 50, 'BMX': 51, 'Playing kickball': 52, 'Shoveling snow': 53, 'Cheerleading': 54, 'Removing ice from car': 55, 'Calf roping': 56, 'Breakdancing': 57, 'Mooping floor': 58, 'Powerbocking': 59, 'Kite flying': 60, 'Getting a tattoo': 61, 'Cleaning shoes': 62, 'Running a marathon': 63, 'Shaving legs': 64, 'Starting a campfire': 65, 'River tubing': 66, 'Zumba': 67, 'Putting on makeup': 68, 'Playing ten pins': 69, 'Raking leaves': 70, 'Doing karate': 71, 'High jump': 72, 'Futsal': 73, 'Grooming dog': 74, 'Wakeboarding': 75, 'Swinging at the playground': 76, 'Playing lacrosse': 77, 'Archery': 78, 'Playing saxophone': 79, 'Long jump': 80, 'Paintball': 81, 'Tango': 82, 'Rope skipping': 83, 'Throwing darts': 84, 'Roof shingle removal': 85, 'Ping-pong': 86, 'Making a sandwich': 87, 'Tennis serve with ball bouncing': 88, 'Triple jump': 89, 'Skiing': 90, 'Peeling potatoes': 91, 'Doing step aerobics': 92, 'Building sandcastles': 93, 'Elliptical trainer': 94, 'Baking cookies': 95, 'Rock-paper-scissors': 96, 'Playing piano': 97, 'Snowboarding': 98, 'Preparing pasta': 99, 'Croquet': 100, 'Playing guitarra': 101, 'Cleaning windows': 102, 'Skateboarding': 103, 'Playing squash': 104, 'Polishing shoes': 105, 'Smoking a cigarette': 106, 'Installing carpet': 107, 'Using the balance beam': 108, 'Drum corps': 109, 'Playing polo': 110, 'Hammer throw': 111, 'Baton twirling': 112, 'Doing crunches': 113, 'Tai chi': 114, 'Kayaking': 115, 'Doing a powerbomb': 116, 'Grooming horse': 117, 'Using the pommel horse': 118, 'Belly dance': 119, 'Clipping cat claws': 120, 'Putting in contact lenses': 121, 'Playing ice hockey': 122, 'Tug of war': 123, 'Brushing hair': 124, 'Welding': 125, 'Mixing drinks': 126, 'Washing hands': 127, 'Having an ice cream': 128, 'Chopping wood': 129, 'Plataform diving': 130, 'Layup drill in basketball': 131, 'Clean and jerk': 132, 'Hitting a pinata': 133, 'Snow tubing': 134, 'Decorating the Christmas tree': 135, 'Pole vault': 136, 'Washing face': 137, 'Hand car wash': 138, 'Doing kickboxing': 139, 'Fixing the roof': 140, 'Dodgeball': 141, 'Playing pool': 142, 'Assembling bicycle': 143, 'Shuffleboard': 144, 'Curling': 145, 'Bullfighting': 146, 'Cricket': 147, 'Snatch': 148, 'Disc dog': 149, 'Fixing bicycle': 150, 'Javelin throw': 151, 'Playing accordion': 152, 'Bathing dog': 153, 'Washing dishes': 154, 'Playing racquetball': 155, 'Shaving': 156, 'Shot put': 157, 'Drinking coffee': 158, 'Hanging wallpaper': 159, 'Springboard diving': 160, 'Ballet': 161, 'Rock climbing': 162, 'Ironing clothes': 163, 'Drinking beer': 164, 'Blowing leaves': 165, 'Using the monkey bar': 166, 'Trimming branches or hedges': 167, 'Fun sliding down': 168, 'Playing field hockey': 169, 'Getting a haircut': 170, 'Cumbia': 171, 'Hula hoop': 172, 'Waterskiing': 173, 'Carving jack-o-lanterns': 174, 'Doing nails': 175, 'Cutting the grass': 176, 'Sumo': 177, 'Making a cake': 178, 'Painting fence': 179, 'Using the rowing machine': 180, 'Brushing teeth': 181, 'Using uneven bars': 182, 'Applying sunscreen': 183, 'Making a lemonade': 184, 'Painting furniture': 185, 'Painting': 186, 'Putting on shoes': 187, 'Volleyball': 188, 'Rollerblading': 189, 'Knitting': 190, 'Polishing forniture': 191, 'Making an omelette': 192, 'Playing rubik cube': 193, 'Cleaning sink': 194, 'Bungee jumping': 195, 'Slacklining': 196, 'Table soccer': 197, 'Waxing skis': 198, 'Laying tile': 199}
            self.ground_truth = self.filt()
        valid_labels = set(self.ground_truth["label"].unique())
            # self.prediction = prediction
        self.prediction = self.prediction[self.prediction["video-id"].isin(self.ground_truth["video-id"])]
        self.a_index = {k: v for k, v in activity_index.items() if v in valid_labels}
            # print("filted activity index: ", self.a_index)

            # Remap a_index to contiguous integers
        label_map = {k: i for i, k in enumerate(self.a_index.keys())}
        self.a_index = label_map

            # Map prediction["label"] to new indices
        inv_old_index = {v: k for k, v in activity_index.items()}
        self.prediction["label"] = self.prediction["label"].map(lambda x: label_map.get(inv_old_index.get(x), -1))
        self.prediction = self.prediction[self.prediction["label"] != -1]
        self.ground_truth["label"] = self.ground_truth["label"].map(lambda x: label_map.get(inv_old_index.get(x), -1))
        self.ground_truth = self.ground_truth[self.ground_truth["label"] != -1]


        print("remapped activity index: ", self.a_index)
        self.activity_index = self.a_index
    
    def filt(self):
        # filter out the ground truth that is not in the prediction
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        # print(self.filt_gt)
        for idx, row in self.ground_truth.iterrows():
            if row["video-id"] in self.filt_gt:
                gt_start = row["t-start"]
                gt_end = row["t-end"]
                # pred_start = self.filt_gt[row["video-id"]]/30.
                # pred_end = self.filt_gt[row["video-id"]]/29. + 106
                # if (gt_start >= pred_start) and (gt_end <= pred_end):
                video_lst.append(row["video-id"])
                t_start_lst.append(row["t-start"])
                t_end_lst.append(row["t-end"])
                label_lst.append(row["label"])
        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth
        

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            if videoid in self.blocked_videos:
                continue

            # remove duplicated instances following ActionFormer
            v_anno = remove_duplicate_annotations(v["annotations"])

            for ann in v_anno:
                if ann["label"] not in activity_index:
                    activity_index[ann["label"]] = cidx
                    cidx =cidx + 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        # if prediction_filename is a string, then json load
        if isinstance(prediction_filename, str):
            with open(prediction_filename, "r") as fobj:
                data = json.load(fobj)
        elif isinstance(prediction_filename, dict):
            data = prediction_filename
        else:
            raise IOError(f"Type of prediction file is {type(prediction_filename)}.")

        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for video_id, v in data["results"].items():
            if video_id in self.blocked_videos:
                continue
            for result in v:
                try:
                    label = self.activity_index[result["label"]]
                except:
                    # this is because the predicted label is not in annotation
                    # such as the some classes only exists in train split, but not in val split
                    label = len(self.activity_index)
                video_lst.append(video_id)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                score_lst.append(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def wrapper_compute_average_precision(self, cidx_list):
        """Computes average precision for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.mAP_result_dict[cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )

    def wrapper_compute_topkx_recall(self, cidx_list):
        """Computes Top-kx recall for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.recall_result_dict[cidx] = compute_topkx_recall_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            )

    def multi_thread_compute_average_precision(self):
        self.mAP_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_average_precision,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = self.mAP_result_dict[i]
        return ap

    def multi_thread_compute_topkx_recall(self):
        self.recall_result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_activity_per_thread = num_total // self.thread + 1

        processes = []
        for tid in range(self.thread):
            num_start = int(tid * num_activity_per_thread)
            num_end = min(num_start + num_activity_per_thread, num_total)

            p = mp.Process(
                target=self.wrapper_compute_topkx_recall,
                args=(list(self.activity_index.values())[num_start:num_end],),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index.items())))
        for i, cidx in enumerate(self.activity_index.values()):
            recall[..., cidx] = self.recall_result_dict[i]
        return recall

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.multi_thread_compute_average_precision()
        self.mAPs = self.ap.mean(axis=1)
        self.average_mAP = self.mAPs.mean()

        metric_dict = dict(average_mAP=self.average_mAP)
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            metric_dict[f"mAP@{tiou}"] = mAP

        # if top_k is not None, we will compute top-kx recall
        if self.top_k is not None:
            self.recall = self.multi_thread_compute_topkx_recall()
            self.mRecall = self.recall.mean(axis=2)

            for tiou, mRecall in zip(self.tiou_thresholds, self.mRecall):
                for k, recall in zip(self.top_k, mRecall):
                    metric_dict[f"recall@{tiou}@{k}"] = recall

        return metric_dict

    def logging(self, logger=None):
        if logger == None:
            pprint = print
        else:
            pprint = logger.info

        pprint("Loaded annotations from {} subset.".format(self.subset))
        pprint("Number of ground truth instances: {}".format(len(self.ground_truth)))
        pprint("Number of predictions: {}".format(len(self.prediction)))
        pprint("Fixed threshold for tiou score: {}".format(self.tiou_thresholds))
        pprint("Average-mAP: {:>4.2f} (%)".format(self.average_mAP * 100))
        for tiou, mAP in zip(self.tiou_thresholds, self.mAPs):
            pprint("mAP at tIoU {:.2f} is {:>4.2f}%".format(tiou, mAP * 100))

        # if top_k is not None, print top-kx recall
        if self.top_k is not None:
            pprint("Fixed top-kx results: {}".format(self.top_k))
            for tiou, recall in zip(self.tiou_thresholds, self.mRecall):
                recall_string = ["R{:d} is {:>4.2f}%".format(k, r * 100) for k, r in zip(self.top_k, recall)]
                pprint("Recall at tIoU {:.2f}: {}".format(tiou, ", ".join(recall_string)))


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    npos = 1 if npos == 0 else npos
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(float)
        this_fp = np.cumsum(fp[tidx, :]).astype(float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)
    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")
    prediction_gbvn = prediction.groupby("video-id")

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts = n_gts + len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred["score"].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[: max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(
            this_pred[["t-start", "t-end"]].values[top_kx_idx], this_gt[["t-start", "t-end"]].values
        )

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[: k * len(this_gt)]
                tp[tidx, kidx] = tp[tidx, kidx] + ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union.clip(1e-8)
    return tIoU


def k_segment_iou(target_segments, candidate_segments):
    return np.stack([segment_iou(target_segment, candidate_segments) for target_segment in target_segments])


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap