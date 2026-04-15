import pytest
from train import get_returns
import torch

def test_returns():
    rewards = [1, 2, 3]
    returns = get_returns(rewards, discount_factor=1)
    assert (returns == torch.tensor([6, 5, 3]).float()).all()