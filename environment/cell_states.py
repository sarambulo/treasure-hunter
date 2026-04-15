from aim_fsm import WallObj, BarrelObj, WorldObject

cell_state_encoding = {"unseen": -1, "empty": 0, "robot": 1, WallObj: 2, BarrelObj: 3, }
cell_state_decoding = {v:k for k,v in cell_state_encoding.items()}

def encode_cell(type):
    if type in cell_state_encoding:
        return cell_state_encoding[type]
    else:
        raise ValueError("Unsupported cell type: %s" % type)

def decode_cell(value):
    if value in cell_state_decoding:
        return cell_state_decoding[value]
    else:
        raise ValueError("Unsupported cell value: %s" % value)