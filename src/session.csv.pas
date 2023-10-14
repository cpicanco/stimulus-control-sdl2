{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, csvdocument, session.csv.enumerator;

type

  { TCSVRows }

  TCSVRows = class(IEnumerable)
  private
    FCSVDocument: TCSVDocument;
  public
    constructor Create;
    destructor Destroy; override;
    function GetEnumerator: IEnumerator;
    procedure LoadFromFile(AFilename : string);
    procedure Clear;
  end;

  function InsideBaseFolder(AFilename : string) : string;
  function InsideBlocksSubFolder(AFilename : string) : string;
  function InsideInstructionsSubFolder(AFilename : string) : string;
  function BaseFileExists(AFilename : string) : Boolean;
  function BlocksFileExists(AFilename : string) : Boolean;
  function InstructionsFileExist(AFilename : string) : Boolean;

implementation

uses session.pool, LazFileUtils;

const
  LDefaultExtention = '.csv';
  LDefaultFolder = 'design';
  LDefaultInstructionsFolder = 'instructions';
  LDefaultBlocksFolder = 'blocks';

function InsideBaseFolder(AFilename: string): string;
begin
  Result := Pool.BaseFilePath +
    LDefaultFolder+DirectorySeparator+
    AFilename+LDefaultExtention;
end;

function InsideBlocksSubFolder(AFilename: string): string;
begin
  Result := Pool.BaseFilePath +
            LDefaultFolder+DirectorySeparator+
            LDefaultBlocksFolder+DirectorySeparator+
            AFilename+LDefaultExtention;
end;

function InsideInstructionsSubFolder(AFilename: string): string;
begin
  Result := Pool.BaseFilePath +
            LDefaultFolder+DirectorySeparator+
            LDefaultInstructionsFolder+DirectorySeparator+
            AFilename+LDefaultExtention;
end;

function BaseFileExists(AFilename: string): Boolean;
begin
  Result := FileExistsUTF8(InsideBaseFolder(AFilename));
end;

function BlocksFileExists(AFilename: string): Boolean;
begin
  Result := FileExistsUTF8(InsideBlocksSubFolder(AFilename));
end;

function InstructionsFileExist(AFilename: string): Boolean;
begin
  Result := FileExistsUTF8(InsideInstructionsSubFolder(AFilename));
end;

constructor TCSVRows.Create;
begin
  inherited Create;
  FCSVDocument := TCSVDocument.Create;
end;

destructor TCSVRows.Destroy;
begin
  FCSVDocument.Free;
  inherited Destroy;
end;

function TCSVRows.GetEnumerator: IEnumerator;
begin
  // TCSVRowParser is reference counted
  Result := TCSVRowParser.Create(FCSVDocument) as IEnumerator;
end;

procedure TCSVRows.LoadFromFile(AFilename: string);
begin
  FCSVDocument.LoadFromFile(AFilename);
end;

procedure TCSVRows.Clear;
begin
  FCSVDocument.Clear;
end;

end.
