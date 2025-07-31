import numpy as np
import scipy.stats as stats

from constants import participants_to_ignore

from explorer import(
    change_data_folder_name,
    foldername_1,
    foldername_2,
    foldername_3)

from fileutils import cd, list_data_folders

from metadata import Metadata

def collect_metadata(foldernames):
    container=[]
    for foldername in foldernames:
        change_data_folder_name(foldername)
        # data_dir()

        for participant in list_data_folders():
            if participant in participants_to_ignore:
                continue
            cd(participant)
            try:
                cd('analysis')
            except FileNotFoundError:
                cd('..')
                continue

            metadata = Metadata()
            # check if key done: exists in metadata.items
            if 'done' in metadata.items.keys():
                if metadata.items['done']:
                    container.append(metadata)
            cd('..')
            cd('..')

    return container

if __name__ == '__main__':
    ages = []
    genders = []
    fields = []
    for metadata in collect_metadata([foldername_1, foldername_2,foldername_3]):
        data = metadata.items
        # if data['code'] == 'VRA':
        #     continue

        # if data['code'] == 'NUN':
        #     continue

        genders.append(data['sex'])

        low, high = data['age'].split('-')
        average = (int(low) + int(high)) / 2
        ages.append(average)

        # print(f'{data["id"]}-{data["code"]}:{data["field"]}: {data["generalization"]}')
        fields.append(data['field'])

    print(f'Participants: {len(ages)}')

    print(f'Females: {genders.count("F")}')
    print(f'Males: {genders.count("M")}')

    
    # Calculate sample mean and standard error of the mean (SEM)
    mean = np.mean(ages)
    sem = stats.sem(ages)

    # Define confidence level (e.g., 95%)
    confidence = 0.95
    df = len(ages) - 1  # Degrees of freedom

    # Get the t-critical value (two-tailed for 95% CI)
    t_critical = stats.t.ppf((1 + confidence) / 2, df)

    # Calculate the margin of error
    margin_of_error = t_critical * sem

    # Confidence interval
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    print(f'Average age: {np.average(ages)}')
    print(f'Min age: {np.min(ages)}')
    print(f'Max age: {np.max(ages)}')
    print(f'CI age: {confidence_interval}')
    print(f'SD age: {np.std(ages)}')