import os

import pandas as pd
import numpy as np
from odf.opendocument import OpenDocumentSpreadsheet, Element
from odf.table import Table, TableRow, TableCell
from odf.text import P
from odf.style import Style, TableCellProperties, TextProperties, Map
from odf.namespaces import OFFICENS, STYLENS, FONS

from odf_custom import CalcElement, conditional_formats_key

folder1_name = 'geovana'
folder2_name = 'rafael'
folder3_name = 'concordancia'
# get script path
script_path = os.path.dirname(os.path.realpath(__file__))
# set current directory to script folder
os.chdir(script_path)

folder1 = os.path.join(script_path, folder1_name)
folder2 = os.path.join(script_path, folder2_name)
folder3 = os.path.join(script_path, folder3_name)

# Helper functions for cell creation
def create_text_cell(text):
    # Create cell with explicit string type
    cell = TableCell(valuetype='string')

    # Add the text element
    p = P(text=text)
    cell.addElement(p)

    return cell

def create_formula_cell(value, formula):
    """Create formula cell with precomputed value"""
    cell = TableCell(
        valuetype='float',
        value=str(value),  # Precomputed value as string
        formula=formula
    )
    cell.addElement(P(text=str(value)))  # Visible text
    return cell

def process_concordance():
    '''
    For each file in folder1
    Read folder1/file
    Read folder2/file
    Add columns from folder1/file to folder2/file
    Save to folder3/file
    '''
    for file in os.listdir(folder1):
        if file.endswith('.data'):
            # Precompute concordance values
            col1_base = f'Speech'
            col2_base = f'Speech-2'
            col1_folder1 = f'Speech_{folder1_name}'
            col1_folder2 = f'Speech_{folder2_name}'
            col2_folder1 = f'Speech-2_{folder1_name}'
            col2_folder2 = f'Speech-2_{folder2_name}'

            dataframe1 = pd.read_csv(os.path.join(folder1, file), sep='\t', encoding='utf-8')
            dataframe2 = pd.read_csv(os.path.join(folder2, file), sep='\t', encoding='utf-8')
            dataframe2.rename(columns={col1_base: col1_folder2, col2_base: col2_folder2}, inplace=True)
            dataframe2[col1_folder1] = dataframe1[col2_base]
            dataframe2[col2_folder1] = dataframe1[col2_base]
            dataframe2.drop(columns=['Speech-3'], inplace=True)


            concordance1 = (
                dataframe2[col1_folder1].astype(str).str.strip().str.lower()
                == dataframe2[col1_folder2].astype(str).str.strip().str.lower()
            ).astype(int).tolist()

            concordance2 = (
                dataframe2[col2_folder1].astype(str).str.strip().str.lower()
                == dataframe2[col2_folder2].astype(str).str.strip().str.lower()
            ).astype(int).tolist()

            # Compute averages
            avg1 = np.mean(concordance1) if concordance1 else 0
            avg2 = np.mean(concordance2) if concordance2 else 0

            # Generate ODS with precomputed values
            doc = OpenDocumentSpreadsheet()

            # Create the "Bad" style in office:styles
            bad_style = Style(
                name="Bad",
                family="table-cell",
                parentstylename="Status"
            )
            # Add table cell properties
            bad_tc_props = TableCellProperties(
                backgroundcolor="#ffcccc",
                wrapoption="no-wrap",
                shrinktofit="false"
            )
            # Add text properties
            bad_text_props = TextProperties(
                color="#cc0000"
            )
            bad_style.addElement(bad_tc_props)
            bad_style.addElement(bad_text_props)
            doc.styles.addElement(bad_style)

            # Create automatic styles in office:automatic-styles
            def create_conditional_style(name):
                style = Style(
                    name=name,
                    family="table-cell",
                    parentstylename="Default"
                )
                style_map = Map(
                    condition="cell-content()=0",
                    applystylename="Bad",
                    basecelladdress="Sheet1.AM2"
                )
                style.addElement(style_map)
                return style

            # Add conditional styles ce1 and ce2
            doc.automaticstyles.addElement(create_conditional_style("ce1"))
            doc.automaticstyles.addElement(create_conditional_style("ce2"))

            # if '1_BAL24' in file:
            #     with open(os.path.join(script_path, folder3, file+'.fods'), 'w', encoding='utf-8') as f:
            #         doc.automaticstyles.toXml(0, f)

            table = Table(name='Sheet1')

            # Header row (unchanged)
            header_row = TableRow()
            for i, col in enumerate(dataframe2.columns):
                cell = TableCell(valuetype='string')
                cell.addElement(P(text=col))
                header_row.addElement(cell)
                if i == len(dataframe2.columns)-1:
                    # Add concordance headers
                    header_row.addElement(create_text_cell('Speech_concordance'))
                    header_row.addElement(create_text_cell('Speech-2_concordance'))
            table.addElement(header_row)

            # Data rows with formula AND precomputed value
            for i, row in dataframe2.iterrows():
                data_row = TableRow()
                for val in row:
                    cell = TableCell(valuetype="string")
                    cell.addElement(P(text=str(val)))
                    data_row.addElement(cell)

                # Add concordance cells with FORMULA + PRECOMPUTED VALUE
                formula1 = f'of:=IF(LOWER(TRIM(AH{i+2}))=LOWER(TRIM(AJ{i+2}));1;0)'
                data_row.addElement(create_formula_cell(concordance1[i], formula1))

                formula2 = f'of:=IF(LOWER(TRIM(AI{i+2}))=LOWER(TRIM(AK{i+2}));1;0)'
                data_row.addElement(create_formula_cell(concordance2[i], formula2))

                table.addElement(data_row)


            # Averages row
            avg_row = TableRow()
            for _ in dataframe2.columns:
                avg_row.addElement(create_text_cell(''))

            avg_formula1 = f'of:=SUM(AL2:AL{len(dataframe2)+1})/COUNT(AL2:AL{len(dataframe2)+1})'
            avg_row.addElement(create_formula_cell(avg1, avg_formula1))

            avg_formula2 = f'of:=SUM(AM2:AM{len(dataframe2)+1})/COUNT(AM2:AM{len(dataframe2)+1})'
            avg_row.addElement(create_formula_cell(avg2, avg_formula2))

            table.addElement(avg_row)


            # <calcext:conditional-formats>
            #  <calcext:conditional-format calcext:target-range-address="Sheet1.AM2:Sheet1.AM59">
            #   <calcext:condition calcext:apply-style-name="Bad" calcext:value="=0" calcext:base-cell-address="Sheet1.AM2"/>
            #  </calcext:conditional-format>
            # </calcext:conditional-formats>


            # Create conditional formatting elements
            conditional_formats = CalcElement('conditional-formats')

            conditional_format = CalcElement('conditional-format')
            conditional_format.set_calcext_attribute('target-range-address', 'Sheet1.AM2:Sheet1.AM59')

            condition = CalcElement('condition')
            condition.set_calcext_attribute('apply-style-name', 'Bad')
            condition.set_calcext_attribute('value', '=0')
            condition.set_calcext_attribute('base-cell-address', 'Sheet1.AM2')

            conditional_format.addElement(condition)
            conditional_formats.addElement(conditional_format)

            if conditional_formats_key not in table.allowed_children:
                table.allowed_children = tuple([c for c in table.allowed_children] + [conditional_formats_key])
            table.addElement(conditional_formats)

            doc.spreadsheet.addElement(table)
            doc.save(os.path.join(folder3, file + '.ods'))

def copy_processed_data_to_parent_folder():
    # get parent foldername
    script_path_parent = os.path.dirname(script_path)
    for file in os.listdir(script_path):
        if file.endswith('.ods'):
            # convert to tsv
            df = pd.read_excel(file, engine='odf', sheet_name='Sheet1', skipfooter=1, header=0)

            df['Speech-2'] = ''

            print(df.columns)

            df.drop(columns=[
                'Speech_rafael',
                'Speech-2_rafael'

            ], inplace=True)

            df.rename(columns={
                'Speech_geovana':'Speech',
                'Speech-2_geovana':'Speech-3'
            })

            filename = os.path.join(script_path_parent, file.replace('.ods', ''))
            df.to_csv(filename, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    # First, calculate concordance from files with trascriptions (from independent observers)
    process_concordance() # *comment or uncomment this line, if needed*

    ##############################################################
    # Now, you can inspect and edit the processed data.ods files #
    # Then, run this script again uncommenting the copy function #
    # to copy the edited files to the parent folder.                                         #
    ##############################################################

    # copy_processed_data_to_parent_folder() # *comment or uncomment this line, if needed*