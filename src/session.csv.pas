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
    constructor Create(AFilename : string; AIsBlockFile : Boolean = False);
    destructor Destroy; override;
    function GetEnumerator: IEnumerator;
  end;

  function BlocksFileExists(AFilename : string) : Boolean;

implementation

uses session.pool, LazFileUtils;

const
  LDefaultExtention = '.csv';
  LDefaultFolder = 'design';
  LDefaultBlocksFolder = 'blocks';

function BlocksFileExists(AFilename: string): Boolean;
begin
  Result := FileExistsUTF8(
    Pool.BaseFilePath +
    LDefaultFolder+DirectorySeparator+
    LDefaultBlocksFolder+DirectorySeparator+
    AFilename+LDefaultExtention);
end;

constructor TCSVRows.Create(AFilename: string; AIsBlockFile: Boolean);
begin
  inherited Create;
  FCSVDocument := TCSVDocument.Create;
  if AIsBlockFile then begin
    FCSVDocument.LoadFromFile(
      LDefaultFolder+DirectorySeparator+
      LDefaultBlocksFolder+DirectorySeparator+
      AFilename+LDefaultExtention);
  end else begin
    FCSVDocument.LoadFromFile(
      LDefaultFolder+DirectorySeparator+AFilename+LDefaultExtention);
  end;
end;

destructor TCSVRows.Destroy;
begin
  FCSVDocument.Free;
end;

function TCSVRows.GetEnumerator: IEnumerator;
begin
  // TCSVRowParser is reference counted
  Result := TCSVRowParser.Create(FCSVDocument) as IEnumerator;
end;

end.
