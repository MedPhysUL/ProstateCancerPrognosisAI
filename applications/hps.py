from src.models.mlp import MLPHP
from src.utils.hyperparameters import Range

MLP_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.MIN: 0.0001,
        Range.MAX: 0.01
    },
    MLPHP.BATCH_SIZE.name: {
        # Range.MIN: 5,
        # Range.MAX: 15,
        # Range.STEP: 5
        Range.VALUE: 2
    },
    MLPHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 0.05
    },
    MLPHP.DROPOUT.name: {
        Range.MIN: 0,
        Range.MAX: 0.25
    },
    MLPHP.LR.name: {
        Range.MIN: 0.0001,
        Range.MAX: 0.01
    },
    MLPHP.RHO.name: {
        Range.VALUE: 0
    },
    MLPHP.N_LAYER.name: {
        Range.VALUE: 3,
    },
    MLPHP.N_UNIT.name: {
        # Range.VALUE: 100,
        Range.VALUE: 50
    },
}
