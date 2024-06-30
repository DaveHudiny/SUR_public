# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: Solution.py file of our SUR project. This file includes Solution abstract class for creation interface of models.
# Topic: Recognition of speaker by images and short voice records.

from abc import ABC, abstractmethod


class Solution(ABC):
    """Abstract class for voice and photo model.

    Args:
        ABC: Some abstract class syntax sugar.
    """
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def fit():
        pass
