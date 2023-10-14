program gird_conversion;

uses Objects, sdl.app.grids.types, sdl.app.grids.methods;

const
  Row = 0;
  Col = 1;

function PositionToCell(AN: Cardinal; AGridSize: Cardinal): TCell;
begin
  Result[Row] := AN div AGridSize;  // Row
  Result[Col] := AN mod AGridSize;  // Column
end;

//function PositionFromObject(AObject: TObject): integer;
//var
//  i: Integer;
//  Cell: TCell;
//begin
//  Result := -1;
//  for i := 0 to FCellsCount - 1 do
//  begin
//    Cell := IntToCell(i, FSeed);
//    if FGrid[Cell[0], Cell[1]].Item = AObject then
//    begin
//      Result := i;
//      Exit;
//    end;
//  end;
//end;

var
  FGrid : TMatrix;
  i : integer;
  Cell : TCell;
  Positions , N: integer;

begin
  InitMonitor;
  N := 3;
  Positions := (N*N);
  for i := 0 to Positions-1 do begin
    Cell := PositionToCell(i, N);
    WriteLn('Position:', i, ', ', 'Row:', Cell[Row], ', ', 'Col:', Cell[Col]);
  end;
  //FGrid := GetCentralGrid(3, 100, False);
  ReadLn;
end.

