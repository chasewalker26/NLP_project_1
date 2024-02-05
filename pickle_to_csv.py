import csv
import pickle
import argparse

def pickle_to_csv(file_name, is_humorous):
        '''
        Converts a .pickle file to a .csv file.

        Inputs:
        file_name:  The CSV file to be converted to a pickle file.
        is_humorous: Whether or not the file is a joke-based file.

        Outputs:
        None
        '''
        
        # Open the specified file
        file = open(file_name, 'rb')
        text = pickle.load(file)
        # Semi-colons were not being handled correctly by
        # the CSV writer, so I switched them out for commas
        for i in range(len(text)):
              if ';' in text[i]:
                    text[i] = text[i].replace(';', ',')
        file.close()

        # Create the labels for the text
        labels = [is_humorous] * len(text)
        
        # Insert the column specifier at the front of the list
        text.insert(0, 'text')
        labels.insert(0, 'humor')

        # Combine the text and labels so that they are
        # in (text, label) pairs for CSV formatting
        to_csv = list(zip(*(text, labels)))
        
        # Change the file type to CSV in the file name string
        csv_name = file_name[:-6] + 'csv'
        # Write the correctly formatted data to the CSV
        csv_file = open(csv_name, 'w')
        writer = csv.writer(csv_file)
        writer.writerows(to_csv)
        csv_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a .pickle file to a .csv file.')
    parser.add_argument('--file_name',
                        type=str,
                        help='The name of the file to be converted.',
                        required=True)
    parser.add_argument('--is_humorous',
                        type=bool,
                        help='Whether or not the file contains jokes.',
                        required=True)
    ARGS, _ = parser.parse_known_args()

    pickle_to_csv(ARGS.file_name, ARGS.is_humorous)
