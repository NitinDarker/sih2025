# from scrape import run_scrape
# from extract import run_extract
# from cleaning_text import run_cleaning
# from llama_local import run_llama
# from llama import run_llama
from extract import  run_extract
# from extract_and_classify import run_extract;


def main():
    print("Starting pipeline...")
    # run_scrape()
    run_extract()
    # run_cleaning()
    # run_llama()
    print("All tasks completed")

if __name__ == "__main__":
    main()
