import os
import importlib
import logging
import sys

import torch
import numpy as np

POINTNET_ROOT = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../../../pointnet/models"
)

MODEL = None
MODEL_KWARGS = {
    "part_num": 2,
    "class_num": 8,
}
CLASS_LABELS = {
    "02691156": 0,
    "02876657": 1,
    "02958343": 2,
    "03001627": 3,
    "03593526": 4,
    "03797390": 5,
    "04256520": 6,
    "04379243": 7,
}
try:
    POINTNET_PATH = os.environ["POINTNETPATH"]
except KeyError:
    pass


def load_pointnet():
    global MODEL
    if MODEL is None:
        logging.debug("Appending pointnet dir to path: {}".format(POINTNET_ROOT))
        sys.path.append(POINTNET_ROOT)
        sys.path.append(os.path.dirname(POINTNET_ROOT))

        model_name = os.listdir(
            os.path.join(os.path.dirname(os.path.dirname(POINTNET_PATH)), "logs")
        )[0].split(".")[0]

        # Load model architecture
        logging.debug("Loading pointnet model {}".format(model_name))
        model_getter = importlib.import_module(model_name)
        MODEL = model_getter.get_model(
            normal_channel=True,
            **MODEL_KWARGS,
        ).cuda()
        MODEL = torch.nn.DataParallel(MODEL)

        # Load checkpoint
        logging.debug("Loading pointnet weights from: {}".format(POINTNET_PATH))
        MODEL.load_state_dict(torch.load(POINTNET_PATH)["model_state_dict"])

        # Set to evaluation mode
        MODEL = MODEL.eval()
    return MODEL


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[
        y.cpu().data.numpy(),
    ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def predict(points, normals, label, num_votes=3):
    """Wrapper for input conversion and voting"""

    def vote(classifier, points, label, class_num, part_num, num_votes=3):
        assert points.size()[0] == 1, "Doesn't support batching"

        vote_pool = torch.zeros(1, points.size()[1], part_num).cuda()
        points = points.transpose(2, 1)

        for _ in range(num_votes):
            seg_pred, _ = classifier(points, to_categorical(label, class_num))
            vote_pool += seg_pred

        seg_pred = vote_pool / num_votes

        return np.argmax(seg_pred.squeeze(0).cpu().data.numpy(), axis=1)

    # Reshape the inputs
    points = torch.from_numpy(
        np.expand_dims(
            np.hstack((points, normals)),
            axis=0,
        )
    ).type(torch.float)
    label = CLASS_LABELS[label]
    label = torch.from_numpy(
        np.expand_dims(np.expand_dims(label, axis=0), axis=0)
    ).type(torch.float)
    classifier = load_pointnet()

    return vote(
        classifier=classifier,
        points=points,
        label=label,
        num_votes=num_votes,
        **MODEL_KWARGS,
    )
