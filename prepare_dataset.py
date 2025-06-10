import pandas as pd
import argparse



def prepaper_dataset(taxonomy_file, dataset, output_file="dataset_cleaned.csv"):
    df = pd.read_csv(dataset)
    label_df = pd.read_csv(taxonomy_file)
    codes = label_df["code"].to_list()
    new_df = df["terms"].str.replace("[",'').str.replace("]","").str.replace("'","").str.replace(" ","").str.get_dummies(",")
    final_df = pd.concat([df,new_df], axis=1)

    def concatenate_title_summary(val1, val2):
        return f"Title: {val1}. Summary: {val2}"

    final_df['title_and_summary'] = final_df.apply(lambda row: concatenate_title_summary(row['titles'], row['summaries']), axis=1)

    final_df
    final_df.to_csv(output_file, index=False)

if __name__=="__main__":
    # Required positional argument
    parser = argparse.ArgumentParser(description='Arguments for The script')

    parser.add_argument('--taxonomy', type=str,
                        help='path to the taxonomy csv file')

    parser.add_argument('--dataset', type=str,
                        help='path to the dataset file')

    # Optional argument
    parser.add_argument('--output_file', type=str,
                        help='path to the output file')

    args = parser.parse_args()
    prepaper_dataset(args.taxonomy, args.dataset, args.output_file)