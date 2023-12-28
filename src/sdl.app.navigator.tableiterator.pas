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
    function Select : ISelectable;
    function NextRow : ISelectable;
    function PreviousRow : ISelectable;
    function NextCol : ISelectable;
    function PreviousCol : ISelectable;
    procedure Update(ASelectables : TSelectables);
    procedure SetBaseControl(ABaseControl : ISelectable);
  end;

implementation

uses sdl.app.output;

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

function TPossibleSelections.Select: ISelectable;
begin
  Result := Iterator.GetCurrent;
end;

function TPossibleSelections.NextRow: ISelectable;
begin
  if Table.RowCount = 1 then begin
    Result := Iterator.GetCurrent;
    Exit;
  end;

  if TableControlExists then begin
    with Iterator do begin
      repeat
        if IsLastRow then begin
          if IsLastCol then begin
            GoFirstCol;
          end else begin
            GoNextCol;
          end;
          GoFirstRow;
        end else begin
          GoNextRow;
        end;
      until not IsCurrentEmpty(Result);
    end;
  end else begin
    if BaseControlExists then begin
      Result := FBaseControl;
    end;
  end;
end;

function TPossibleSelections.PreviousRow: ISelectable;
begin
  if Table.RowCount = 1 then begin
    Result := Iterator.GetCurrent;
    Exit;
  end;

  if TableControlExists then begin
    with Iterator do begin
      repeat
        if IsFirstRow then begin
          if IsFirstCol then begin
            GoLastCol;
          end else begin
            GoPreviousCol;
          end;
          GoLastRow;
        end else begin
          GoPreviousRow;
        end;
      until not IsCurrentEmpty(Result);
    end;
  end else begin
    if BaseControlExists then begin
      Result := FBaseControl;
    end;
  end;
end;

function TPossibleSelections.NextCol: ISelectable;
begin
  if Table.ColCount = 1 then begin
    Result := Iterator.GetCurrent;
    Exit;
  end;

  if TableControlExists then begin
    with Iterator do begin
      repeat
        if IsLastCol then begin
          if IsLastRow then begin
            GoFirstRow;
          end else begin
            GoNextRow;
          end;
          GoFirstCol;
        end else begin
          GoNextCol;
        end;
      until not IsCurrentEmpty(Result);
    end;
  end else begin
    if BaseControlExists then begin
      Result := FBaseControl;
    end;
  end;
end;

function TPossibleSelections.PreviousCol: ISelectable;
begin
  if Table.ColCount = 1 then begin
    Result := Iterator.GetCurrent;
    Exit;
  end;

  if TableControlExists then begin
    with Iterator do begin
      repeat
        if IsFirstCol then begin
          if IsFirstRow then begin
            GoLastRow;
          end else begin
            GoPreviousRow;
          end;
          GoLastCol;
        end else begin
          GoPreviousCol;
        end;
      until not IsCurrentEmpty(Result);
    end;
  end else begin
    if BaseControlExists then begin
      Result := FBaseControl;
    end;
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

  procedure Count(ARows : TPosition; ACols : TPosition; ATable: TTableSpec);
  var
    LSelectable : ISelectable;
    LColCount : UInt32 = 0;
    LRowCount : UInt32 = 0;
    LX : integer;
    LY : integer;
  begin
    for LSelectable in ASelectables do begin
      LX := LSelectable.Origen.x;
      LY := LSelectable.Origen.y;

      with ARows do begin
        if ContainsKey(LY) then begin
          { do nothing }
        end else begin
          Add(LY, LRowCount);
          Inc(LRowCount);
        end;
      end;

      with ACols do begin
        if ContainsKey(LX) then begin
          { do nothing }
        end else begin
          Add(LX, LColCount);
          Inc(LColCount);
        end;
      end;
    end;
    ATable.RowCount := LRowCount;
    ATable.ColCount := LColCount;
    ATable.Populate;
    Print(ATable.AsString);
  end;

begin
  Table.Clear;
  if ASelectables = nil then Exit;

  LRows := TPosition.Create;
  LCols := TPosition.Create;
  try
    Count(LRows, LCols, Table);
    for LISelectable in ASelectables do begin
      LRow := LISelectable.Origen.y;
      LCol := LISelectable.Origen.x;
      Table.Cells[LCols[LCol], LRows[LRow]] := LISelectable;
    end;
  finally
    LRows.Free;
    LCols.Free;
  end;

  Iterator.GoFirstRow;
  Iterator.GoFirstCol;
end;

procedure TPossibleSelections.SetBaseControl(ABaseControl: ISelectable);
begin
  FBaseControl := ABaseControl;
end;

end.

