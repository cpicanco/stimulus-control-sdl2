{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.navigator.tableiterator;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections,
  Generics.Aggregator.Table,
  sdl.app.selectable.contract,
  sdl.app.selectable.list;

type

  { TPossibleSelections }

  TPossibleSelections = class (specialize TTableAggregator<ISelectable>)
  private
    FBaseControl : ISelectable;
    function TableControlExists : Boolean;
    function BaseControlExists : Boolean;
  public
    constructor Create; override;
    function Select : ISelectable;
    function GoBottom : ISelectable;
    function GoTop : ISelectable;
    function GoRight : ISelectable;
    function GoLeft : ISelectable;
    function GoTopRight : ISelectable;
    function GoBottomLeft : ISelectable;
    function GoTopLeft : ISelectable;
    function GoBottomRight : ISelectable;
    function GoBaseControl : ISelectable;
    function GoTarget(AControl: ISelectable) : ISelectable;

    //function TopLeft
    procedure Update(ASelectables : TSelectables);
    procedure SetBaseControl(ABaseControl : ISelectable);
  end;

implementation

uses
  Generics.Tables.Types
  //, sdl.app.output
  , integers.list;

{ TPossibleSelections }

function TPossibleSelections.TableControlExists: Boolean;
begin
  Result := (Table.RowCount > 0) and (Table.ColCount > 0);
end;

function TPossibleSelections.BaseControlExists: Boolean;
begin
  if FBaseControl <> nil then begin
    Result := True;
  end else begin
    Result := False;
    raise EArgumentNilException.Create('Base control = nil');
  end;
end;

constructor TPossibleSelections.Create;
begin
  inherited Create;
  FBaseControl := nil;
end;

function TPossibleSelections.Select: ISelectable;
begin
  Result := Iterator.GetCurrent;
end;

function TPossibleSelections.GoBottom: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while (not IsLastRow) do begin
      GoNextRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoTop: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while (not IsFirstRow) do begin
      GoPreviousRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoRight: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while (not IsLastCol) do begin
      GoNextCol;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoLeft: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while (not IsFirstCol) do begin
      GoPreviousCol;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoTopRight: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while not (IsLastCol or IsFirstRow) do begin
      GoNextCol;
      GoPreviousRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoBottomLeft: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while not (IsFirstCol or IsLastRow) do begin
      GoPreviousCol;
      GoNextRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoTopLeft: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while not (IsFirstCol or IsFirstRow) do begin
      GoPreviousCol;
      GoPreviousRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoBottomRight: ISelectable;
begin
  if (Table.ColCount = 1) and (Table.RowCount = 1) then begin
    Result := Select;
    Exit;
  end;

  Result := nil;
  with Iterator do begin
    Save;
    while not (IsLastCol or IsLastRow) do begin
      GoNextCol;
      GoNextRow;
      if IsCurrentEmpty(Result) then begin
        Result := nil;
      end else begin
        Exit;
      end;
    end;
    Load;
  end;
end;

function TPossibleSelections.GoBaseControl: ISelectable;
begin
  Iterator.GoToCell(Table.CellOf(FBaseControl));
  Result := FBaseControl;
end;

function TPossibleSelections.GoTarget(AControl: ISelectable): ISelectable;
var
  LCell : TCell;
  LTargetFound : Boolean = False;
begin
  LCell := Table.CellOfTarget(AControl, LTargetFound);
  if LTargetFound then begin
    Iterator.GoToCell(LCell);
    Result := AControl;
  end else begin
    Result := nil;
  end;
end;

procedure TPossibleSelections.Update(ASelectables: TSelectables);
var
  LISelectable : ISelectable;
type
  TPosition = specialize TDictionary<Integer, UInt32>;
var
  LRows : TPosition;
  LCols : TPosition;
  LRow : integer;
  LCol : integer;

  procedure MountGrid(ARows : TPosition; ACols : TPosition; ATable: TTableSpec);
  var
    LSelectable : ISelectable;
    LColCount : UInt32 = 0;
    LRowCount : UInt32 = 0;
    LX : integer;
    LY : integer;
    LSortedX : TIntegers;
    LSortedY : TIntegers;
  begin
    LSortedX := TIntegers.Create(TIntegers.ByValue);
    try
      for LSelectable in ASelectables do begin
        LSortedX.Add(LSelectable.Origen.x);
      end;

      LSortedX.Sort;
      for LX in LSortedX do begin
        with ACols do begin
          if ContainsKey(LX) then begin
            { do nothing }
          end else begin
            Add(LX, LColCount);
            Inc(LColCount);
          end;
        end;
      end;
    finally
      LSortedX.Free;
    end;

    LSortedY := TIntegers.Create();
    try
      for LSelectable in ASelectables do begin
        LSortedY.Add(LSelectable.Origen.y);
      end;

      LSortedY.Sort;
      for LY in LSortedY do begin
        with ARows do begin
          if ContainsKey(LY) then begin
            { do nothing }
          end else begin
            Add(LY, LRowCount);
            Inc(LRowCount);
          end;
        end;
      end;
    finally
      LSortedY.Free;
    end;

    ATable.RowCount := LRowCount;
    ATable.ColCount := LColCount;
    ATable.Populate;
  end;

begin
  Table.Clear;
  if ASelectables = nil then Exit;

  LRows := TPosition.Create;
  LCols := TPosition.Create;
  try
    MountGrid(LRows, LCols, Table);
    for LISelectable in ASelectables do begin
      LRow := LISelectable.Origen.y;
      LCol := LISelectable.Origen.x;
      Table.Cells[LCols[LCol], LRows[LRow]] := LISelectable;
    end;
  finally
    LRows.Free;
    LCols.Free;
  end;

  Iterator.GoToCell(Table.CellOf(FBaseControl));
end;

procedure TPossibleSelections.SetBaseControl(ABaseControl: ISelectable);
begin
  FBaseControl := ABaseControl;
end;

end.

