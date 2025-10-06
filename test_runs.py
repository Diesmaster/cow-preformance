from data_processor.DataProcessor import DataProcessing



def main():
    processor = DataProcessing()
    arr = processor.get_variables(3)

    print(f"datapoints: {len(arr)}")

if __name__ == "__main__":
    main()
