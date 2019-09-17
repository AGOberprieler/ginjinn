#!/usr/bin/env python

import sys

from ginjinn import config
from ginjinn.core import parser
from ginjinn.core import Project

def main():
    print(config.PLATFORM)
    print(config.MODELS_PATH, config.RESEARCH_PATH, config.SLIM_PATH)

    args = parser.parse_args()
    print(args)

if __name__ == '__main__':
    main()