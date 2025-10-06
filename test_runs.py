from data_processor.DataProcessor import DataProcessing



def main():
    processor = DataProcessing()

    n_weigings = [3,4,5]

    dfs = processor.get_dfs(n_weigings)

    for df in dfs:
        print(f"DF n = {df}, has {len(dfs[df])} entries")


if __name__ == "__main__":
    main()
