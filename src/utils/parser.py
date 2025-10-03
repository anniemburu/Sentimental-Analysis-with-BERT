import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Pipeline Arguments")
    
    parser.add_argument('-autodownload', '--autodownload', action="store_true", help="Define if the data should be downloaded automatically")

    return parser


#