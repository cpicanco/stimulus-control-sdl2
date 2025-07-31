{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.csv.enumerator;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes, SysUtils, session.csv.document;

type

  { IEnumerator }

  IEnumerator = interface(IInterface)
    ['{1792C9B9-7465-42A2-8298-5E0FCD6783FC}']
    function GetCurrent: TStringList;
    function GetIndexOF(AIndex : integer): TStringList;
    function MoveNext: Boolean;
    procedure Reset;
    property Current: TStringList read GetCurrent;
    property IndexOf[AIndex : integer]: TStringList read GetIndexOF;
  end;

  { IEnumerable }

  IEnumerable = interface
    ['{1377B9CC-479C-433A-BB00-9A25129F01E8}']
    function GetEnumerator: IEnumerator;
  end;

{ TCSVRowParser }

  TCSVRowParser = class(TInterfacedObject, IEnumerator)
  private
    FCSV: TCustomCSVDocument;
    FCurrent : TStringList;
    function GetCurrent: TStringList;
    function GetIndexOf(AIndex : integer): TStringList;
    function MoveNext: Boolean;
    procedure Reset;
  public
    constructor Create(CSV: TCustomCSVDocument);
    destructor Destroy; override;
  end;


implementation

constructor TCSVRowParser.Create(CSV: TCustomCSVDocument);
begin
  inherited Create;
  FCSV := CSV;
  FCurrent := TStringList.Create;
  FCurrent.Sorted := True;
  FCurrent.Duplicates := dupIgnore;
  if FCSV.SkipHeader then begin
    FCSV.CurrentIndex := 0;
  end else begin
    FCSV.CurrentIndex := 0;
  end;
end;

destructor TCSVRowParser.Destroy;
begin
  FCurrent.Clear;
  FCurrent.Free;
  inherited Destroy;
end;

function TCSVRowParser.GetCurrent: TStringList;
var
  LColIndex: Integer;
  LName, LValue : string;
begin
  if (FCSV.RowCount = 0) or
     (FCSV.MaxColCount = 0) then
    raise EInvalidOpException.Create('Empty CSV document.');

  FCurrent.BeginUpdate;
  FCurrent.Clear;
  with FCSV do begin
    if (CurrentIndex >= 0) and (CurrentIndex < RowCount) then
      for LColIndex := 0 to MaxColCount - 1 do begin
        LName := Cells[LColIndex, 0].Trim;
        LValue := Cells[LColIndex, CurrentIndex].Trim;
        if LName.IsEmpty then begin
          { do nothing }
        end else begin
          FCurrent.Values[LName] := LValue;
        end;
      end;
   end;
  FCurrent.EndUpdate;
  Result := FCurrent;
end;

function TCSVRowParser.GetIndexOf(AIndex : integer): TStringList;
begin
  with FCSV do begin
    if (CurrentIndex >= 0) and (CurrentIndex < RowCount) then begin
      FCSV.CurrentIndex := AIndex;
    end else begin
      raise EArgumentOutOfRangeException.Create(
        'Index:'+AIndex.ToString +'; RowCount:'+ RowCount.ToString);
    end;
  end;
  Result := GetCurrent;
end;

function TCSVRowParser.MoveNext: Boolean;
begin
  FCSV.CurrentIndex := FCSV.CurrentIndex+1;
  Result := FCSV.CurrentIndex < FCSV.RowCount;
end;

procedure TCSVRowParser.Reset;
begin
  FCSV.CurrentIndex := 0;
end;

end.

