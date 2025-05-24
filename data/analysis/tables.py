import os

import pandas as pd
from bs4 import BeautifulSoup
import copy
import sass

from words import pre_test_hardcoded_order, cycle_per_word, category_per_word
from fileutils import cd
from speech import target_speech, load_probes_file

# Function to compare two strings and make equal characters bold
def format_words(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        if str2 == 'ns':
            return '-'
        else:
            if str1 == str2:
                return f'<span style="background-color: #A2A2A2;">{str2}</span>'
            else:
                return str2
            # # check if str1 and str2 are the same length
            # if len(str1) != len(str2):
            #     # if not, return a string with the same length as str2 appending - to str1
            #     str1 = str1 + '-' * (len(str2) - len(str1))

            # html bold tag
            # return ''.join(f'<b>{c1}</b>' \
            #     if c1 == c2 else c2 for c1, c2 in zip(str1, str2))

            # html span tag with background color
            # return ''.join(f'<span style="background-color: #A2A2A2;">{c1}</span>' \
            #     if c1 == c2 else c2 for c1, c2 in zip(str1, str2))
    else:
        return '-'

def create_html_tables(df):
    # Create a dictionary to store the DataFrames for each participant
    unique_studies = df['study']
    for study in unique_studies:
        dfs = {}
        # Iterate over the unique participants
        for participant in df['Participant'].unique():
            # Create a new DataFrame for this participant
            df_participant = df[df['Participant'] == participant].copy()

            df_participant.drop(columns='Participant', inplace=True)
            # Create the 'HTML' column
            df_participant['HTML'] = df_participant.apply(lambda row: format_words(row['Name'], row[target_speech]), axis=1)

            condition7 = df_participant[df_participant['Condition'] == 7].copy()
            condition5 = df_participant[df_participant['Condition'] == 5].copy()

            df_participant = condition7
            # join date time columns
            df_participant['Date'] = df_participant['Date'] + ' ' + df_participant['Time']

            # find unique date/time
            dates = df_participant['Date'].unique()
            # get bena and falo data
            bena_falo_tbody, cycle_per_date5 = get_bena_falo_tbody(participant, condition5, dates)
            # create a dictionary to store Cycle.ID of each Date
            if '12-MED' in participant:
                df_participant['Cycle.ID'] = df_participant['Cycle.ID'].replace(0, 6)

            cycle_per_date = {}
            for date in dates:
                cycle_per_date[date] = df_participant[df_participant['Date'] == date]['Cycle.ID'].values[0]

            container = {}
            for date in dates:
                df2 = pd.DataFrame()
                df2['Word'] = df_participant[df_participant['Date'] == date]['Name']
                # use category_per_word to get word categories
                df2['Category'] = df2['Word'].map(category_per_word)
                # category must be the first column
                df2 = df2[['Category', 'Word']]
                # use cycle_per_word to get the cycle of each word
                df2['Cycles'] = df2['Word'].map(cycle_per_word)
                df2['Cycles'] = pd.to_numeric(df2['Cycles'], errors='coerce').fillna(0).astype(int)
                df2[date] = df_participant[df_participant['Date'] == date]['HTML']
                # make sure that the order is the same as the pre_test_hardcoded_order
                df2['Word'] = pd.Categorical(df2['Word'], pre_test_hardcoded_order)
                df2 = df2.sort_values('Word')
                # drop all word columns, except the first one
                if len(container) > 0:
                    df2.drop(columns='Word', inplace=True)
                    df2.drop(columns='Cycles', inplace=True)
                    df2.drop(columns='Category', inplace=True)

                # reset index
                df2.reset_index(drop=True, inplace=True)
                container[date] = df2

            # concatenate all dataframes
            df_participant = pd.concat(container.values(), axis=1)

            # Store the DataFrame in the dictionary
            dfs[participant] = (df_participant, cycle_per_date, bena_falo_tbody, cycle_per_date5)

        # Convert DataFrames to an HTML table using tabulate and save it to a file
        for participant, (df_participant, cycle_per_date, body, cycle_per_date5) in dfs.items():
            # # Add the new column
            # df_participant.insert(0, 'Category', ['Teaching']*12 + ['Assessment']*4 + ['Assessment2']*4)

            # Convert the DataFrame to HTML
            html = df_participant.to_html(escape=False, index=False)

            html_table = BeautifulSoup(html, 'html.parser')
            # use soup to duplicate the first row
            first_row = html_table.select('tr:first-child')[0]
            # insert three rows after the first row
            first_row.insert_after(copy.deepcopy(first_row))
            # iterate over the cells of the second row
            for cell in html_table.select('tr:nth-child(2) th'):
                # use cycle_per_date to get the cycle of each date
                if cell.string in cycle_per_date:
                    cycle = cycle_per_date[cell.string]
                    # insert the cycle number in the cell
                    cell.string = f'Cycle {cycle}'

            # Set the rowspan attribute and remove the spanned cells for header
            for category in ['Category','Word', 'Cycles']:
                cells = html_table.select(f'th:-soup-contains("{category}")')
                cells[0]['rowspan'] = len(cells)
                for cell in cells[1:]:
                    cell.decompose()

            # Select the cells in the first row from the third column onwards
            cells = html_table.select('tr:first-child th:nth-of-type(n+3)')

            # Add classes to the table to identify rows with data from the same cycle
            for thead in html_table.find_all('thead'):
                tr_tags = thead.find_all('tr')
                # get the first one
                th_tags = tr_tags[0].find_all('th')

            # create a new dictionary with counts of cycles
            for tbody in html_table.find_all('tbody'):
                # # bena and falo
                tr_tags = body.find_all('tr')
                for tr in reversed(tr_tags):
                    tbody.insert(0, tr)
                # teaching and assessment
                tr_tags = tbody.find_all('tr')
                for tr in tr_tags:
                    cycle_count_per_date = {}
                    # if tr is odd and less then 12 add class "training-row"
                    if tr_tags.index(tr) % 2 != 0:
                        tr['class'] = 'even-row'
                    else:
                        tr['class'] = 'odd-row'

                    if tr_tags.index(tr) < 4:
                        tr['class'] += f' assessment3-row word-{tr_tags.index(tr)+1}'
                    elif tr_tags.index(tr) >= 4 and tr_tags.index(tr) < 16:
                        tr['class'] += f' training-row word-{tr_tags.index(tr)+1}'
                    elif tr_tags.index(tr) >= 16 and tr_tags.index(tr) < 20:
                        tr['class'] += f' assessment1-row word-{tr_tags.index(tr)+1}'
                    else:
                        tr['class'] += f' assessment2-row word-{tr_tags.index(tr)+1}'

                    td_tags = tr.find_all('td')

                    # add class "label-col-end" to td_tags[2]
                    td_tags[0]['class'] = 'category-col'
                    td_tags[1]['class'] = 'word-col'
                    td_tags[2]['class'] = 'labels-end-col'

                    # content columns start at index 3
                    td_start = 3
                    header = list()
                    for td, th in zip(td_tags[td_start:], th_tags[3:]):
                        if th.string in cycle_per_date:
                            class1 = 'training-col'
                            cycle = cycle_per_date[th.string]

                            # count the number of occurrences of each cycle
                            if cycle in cycle_count_per_date:
                                cycle_count_per_date[cycle] += 1
                            else:
                                cycle_count_per_date[cycle] = 1

                            # if first occurrence of cycle, add "first-occurence" class
                            if cycle_count_per_date[cycle] == 1:
                                class2 = f'cycle{cycle} first-occurrence'
                            else:
                                if cycle == 6:
                                    class2 = f'cycle{cycle} last-occurrence'
                                else:
                                    class2 = f'cycle{cycle}'

                            td['class'] = ' '.join([class1, class2])

            # Replace the white space with a <br> tag
            for cell in cells:
                parts = cell.string.split(' ', 1)
                if len(parts) == 1:
                    continue
                cell.clear()
                cell.append(parts[0])
                cell.append(html_table.new_tag('br'))
                cell.append(parts[1])

            # Set the rowspan attribute and remove the spanned cells
            for category in ['Teaching', 'Generalization1', 'Generalization2']:
                cells = html_table.select(f'td:-soup-contains("{category}")')
                if cells:
                    cells[0]['rowspan'] = len(cells)
                    for cell in cells[1:]:
                        cell.decompose()
                    # Replace the content of the cell if Teaching
                    if category == 'Teaching':
                        cells[0].clear()
                        cells[0].append('Teaching')
                        cells[0].append(html_table.new_tag('br'))
                        cells[0].append('(hits below')
                        cells[0].append(html_table.new_tag('br'))
                        cells[0].append('dashed lines are')
                        cells[0].append(html_table.new_tag('br'))
                        cells[0].append('evidence of')
                        cells[0].append(html_table.new_tag('br'))
                        cells[0].append('generalization)')
                else:
                    print(f'Category {category} not found in participant {participant}')

            # replace Assessment with "Assessment<br>Syllabic|Intrasyllabic"
            cells = html_table.select('td:-soup-contains("Generalization1")')
            if cells:
                br_tag = html_table.new_tag('br')
                cells[0].clear()
                cells[0].append('Generalization')
                cells[0].append(br_tag)
                cells[0].append('(Syllabic|Intrasyllabic)')
            else:
                print(f'Category Generalization1 not found in participant {participant}')

            # replace Generalization with "Assessment<br>Intrasyllabic|Syllabic"
            cells = html_table.select('td:-soup-contains("Generalization2")')
            if cells:
                br_tag = html_table.new_tag('br')
                cells[0].clear()
                cells[0].append('Generalization')
                cells[0].append(br_tag)
                cells[0].append('(Intrasyllabic|Syllabic)')
            else:
                print(f'Category Generalization2 not found in participant {participant}')

            # replace Generalization with "Recombinative<br>(Constant words)"
            cells = html_table.select('td:-soup-contains("Generalization3")')
            if cells:
                cells[0]['rowspan'] = len(cells)
                for cell in cells[1:]:
                    cell.decompose()
                cells[0].clear()
                cells[0].append('Recombinative')
                cells[0].append(html_table.new_tag('br'))
                cells[0].append('reading')
                cells[0].append(html_table.new_tag('br'))
                cells[0].append('(Constant words)')
            else:
                print(f'Category Generalization not found in participant {participant}')


            for category in ['0', '1', '2', '3', '6', '7', 'falo', 'bena']:
                if category == 'falo' or category == 'bena':
                    cells = html_table.select(f'td.word-col:-soup-contains("{category}")')
                else:
                    cells = html_table.select(f'td:-soup-contains("{category}")')
                if cells:
                    cells[0]['rowspan'] = len(cells)
                    for cell in cells[1:]:
                        cell.decompose()
                else:
                    print(f'Category {category} not found in participant {participant}')

            for category in ['4', '5']:
                cells = html_table.select(f'td:-soup-contains("{category}")')
                if cells:
                    cells[0]['rowspan'] = 2
                    cells[2]['rowspan'] = 2
                    for cell in cells[1:2]:
                        cell.decompose()
                    for cell in cells[3:]:
                        cell.decompose()
                else:
                    print(f'Category {category} not found in participant {participant}')

            # replace 0 in Cycles with None
            cells = html_table.select('td:-soup-contains("0")')
            for cell in cells:
                cell.string = 'None'

            # replace 7 in Cycles with All
            cells = html_table.select('td:-soup-contains("7")')
            for cell in cells:
                cell.string = 'All'

            cells = html_table.select('th:-soup-contains("Cycle 6")')
            if len(cells) > 1:
                if cells[-1].string == "Cycle 6":
                    cells[-1].string = 'Final'

            # cells = html_table.select('th:-soup-contains("Cycle 1")')
            # if len(cells) > 0:
            #     if cells[0].string == "Cycle 1":
            #         cells[0].string = 'Pre-test'

            # add a second empty thead after the first one
            first_thead = html_table.select('thead')[0]
            second_thead = html_table.new_tag('thead')
            # get the length of columns
            tr_tags = html_table.select('tr')
            th_tags = tr_tags[0].find_all('th')
            columns = len(th_tags)

            tr_tag = html_table.new_tag('tr')
            th_tag = html_table.new_tag('th')
            th_tag['colspan'] = columns
            th_tag.string = 'CD probes after teaching in day'
            tr_tag.append(th_tag)
            second_thead.append(tr_tag)
            first_thead.insert_after(second_thead)

            # move the first 4 tr tags inside tbody to a second tbody
            second_tbody = html_table.find('tbody')
            first_tbody = html_table.new_tag('tbody')
            tr_tags = second_tbody.find_all('tr')
            for tr in tr_tags[:4]:
                first_tbody.append(tr)

            # append first_tbody before second_tbody
            second_tbody.insert_before(first_tbody)

            third_thead = html_table.new_tag('thead')
            tr_tag = html_table.new_tag('tr')
            th_tag = html_table.new_tag('th')
            th_tag['colspan'] = columns
            th_tag.string = 'CD probes before teaching in day'
            tr_tag.append(th_tag)

            third_thead.append(tr_tag)
            first_tbody.insert_after(third_thead)

            # Create a new BeautifulSoup object
            soup = BeautifulSoup('', 'html.parser')
            html_tag = soup.new_tag('html')
            head_tag = soup.new_tag('head')
            link_tag = soup.new_tag('link', rel='stylesheet', href='style.css')
            body_tag = soup.new_tag('body')
            body_tag.append(html_table)
            head_tag.append(link_tag)
            # Append the table to the body tag
            html_tag.append(head_tag)
            html_tag.append(body_tag)
            soup.clear()
            soup.append(html_tag)

            # Convert the modified HTML back to a string
            html = str(soup)

            # Write the HTML to a file
            participant = participant.replace('\\', '').replace('-', '_')
            filename = f'{study}_probes_CD_{participant}'
            with open(os.path.join('output', 'tables', filename), 'w') as f:
                f.write(html)
            print(f'File {filename} created.')

def get_bena_falo_tbody(participant, df_participant, condition7_dates):
    # find unique dates in condition 5
    dates5 = df_participant['Date'].unique()

    cycle_per_date = {}
    for date in dates5:
        cycle_per_date[date] = df_participant[df_participant['Date'] == date]['Cycle.ID'].values[0]

    # create a new dataframe with only bena and falo
    pattern = r'(bena|falo)'
    df_constant = df_participant[df_participant['Name'].str.match(pattern)].copy()
    # df_other = df_participant[~df_participant['Name'].str.match(pattern)].copy()
    container = {}
    for date7 in condition7_dates:
        df2 = pd.DataFrame()
        # print(dates5)
        date = date7.split(' ')[0]
        if date in dates5:
            # print(f'Trying to build falo/bena column for date {date.split(" ")[0]}')
            if '5-JUL' in participant:
                # get the first 4 rows of each date
                df2 = df_constant[df_constant['Date'] == date].head(4)

            df2['Word'] = df_constant[df_constant['Date'] == date]['Name']
            # use category_per_word to get word categories
            df2['Category'] = df2['Word'].map(category_per_word)
            # category must be the first column
            df2 = df2[['Category', 'Word']]
            # use cycle_per_word to get the cycle of each word
            df2['Cycles'] = df2['Word'].map(cycle_per_word)
            df2['Cycles'] = pd.to_numeric(df2['Cycles'], errors='coerce').fillna(0).astype(int)
            df2[date] = df_constant[df_constant['Date'] == date]['HTML']
            # make sure that the order is the same as the pre_test_hardcoded_order
            df2['Word'] = pd.Categorical(df2['Word'], pre_test_hardcoded_order)
            df2 = df2.sort_values('Word')
        else:
            # print(f'Date {date.split(" ")[0]} does not exist in condition 5. Using hardcoded values for bena and falo.')
            df2['Category'] = ['Generalization3', 'Generalization3', 'Generalization3', 'Generalization3']
            df2['Word'] = ['bena', 'bena', 'falo', 'falo']
            df2['Cycles'] = [7, 7, 7, 7]
            df2[date] = ['', '', '', '']

        if len(container) > 0:
            df2.drop(columns='Word', inplace=True)
            df2.drop(columns='Category', inplace=True)
            df2.drop(columns='Cycles', inplace=True)

        # reset index
        df2.reset_index(drop=True, inplace=True)
        container[date] = df2

    # concatenate all dataframes
    df_participant = pd.concat(container.values(), axis=1)
    # Convert the DataFrame to HTML
    html = df_participant.to_html(escape=False, index=False)
    html_table = BeautifulSoup(html, 'html.parser')
    # use soup to duplicate the first row
    first_row = html_table.select('tr:first-child')[0]
    # insert three rows after the first row
    first_row.insert_after(copy.deepcopy(first_row))
    # iterate over the cells of the second row
    for cell in html_table.select('tr:nth-child(2) th'):
        # use cycle_per_date to get the cycle of each date
        if cell.string in cycle_per_date:
            cycle = cycle_per_date[cell.string]
            # insert the cycle number in the cell
            cell.string = f'Cycle {cycle}'

    # get table tbody
    return html_table.find('tbody'), cycle_per_date

def compile_sass():
    # compile sass to css
    sass.compile(dirname=('.', '.'))

if __name__ == "__main__":
    compile_sass()
    create_html_tables()