#!/usr/bin/env python

from ginjinn import config

def main():
    print('hello from main')
    print(config.PLATFORM)
    print(config.MODELS_PATH, config.RESEARCH_PATH, config.SLIM_PATH)

if __name__ == '__main__':
    main()