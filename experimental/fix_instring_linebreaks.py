import csv

def clean_message(message):
    # Remove line breaks (newlines and carriage returns)
    return message.replace('\n', '').replace('\r', '')

def process_csv(input_filename, output_filename):
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        
        # Create CSV reader and writer
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Initialize CSV writer with the same headers
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

        # Process each row in the CSV file
        cnt = 0
        for row in reader:
            cnt += 1
            print(f'Processing line {cnt} ...')
            # Clean the message field
            row['message'] = clean_message(row['message'])
            # Write the cleaned row to the output file
            writer.writerow(row)

input_file = 'data/gaia/csv/log.csv'
output_file = 'data/gaia/csv/log-fixed.csv'

process_csv(input_file, output_file)

print("Processing complete. Cleaned CSV written to:", output_file)
