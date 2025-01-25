{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Generics.Table;

{$mode objfpc}

interface

uses
  Classes, SysUtils, Generics.Collections, Generics.Tables.Types;

type

  { TTable }

  generic TTable<T> = class
  private
    type
      TTableSpec = specialize TDictionary<UInt64, T>;

    var
      FTableData: TTableSpec;
      FRowCount: UInt32;
      FColCount: UInt32;

      function KeyOf(ACol, ARow: UInt32): UInt64;
      function GetCell(ACol, ARow: UInt32): T;
      procedure SetCell(ACol, ARow: UInt32; const AValue: T);
      function GetRowCount: UInt32;
      function GetColCount: UInt32;
      function IsCellEmpty(const AValue: T): Boolean; virtual; overload;
      procedure SetColCount(AValue: UInt32);
      procedure SetRowCount(AValue: UInt32);
    public
      constructor Create;
      constructor Create(ARowCount, AColCount: UInt32);
      destructor Destroy; override;
      function IsCellEmpty(ACol, ARow: UInt32; out ACell : T) : Boolean; overload;
      function IsCellEmpty(ACol, ARow: UInt32): Boolean; overload;
      function AsString : string;
      function CellOf(AValue : T) : TCell;
      function CellOfTarget(AValue: T; out ATargetFound : Boolean) : TCell;
      procedure AppendRow;
      procedure AppendCol;
      procedure InsertRow(ARow: UInt32; const AValues: array of T);
      procedure InsertCol(ACol: UInt32; const AValues: array of T);
      procedure RemoveRow(ARow: UInt32);
      procedure RemoveCol(ACol: UInt32);

      procedure Clear;
      procedure Empty;
      procedure Populate;
      procedure Invalidate;
      property Cells[ACol, ARow: UInt32]: T read GetCell write SetCell; default;
      property RowCount: UInt32 read GetRowCount write SetRowCount;
      property ColCount: UInt32 read GetColCount write SetColCount;
  end;

implementation

{ TTable }

constructor TTable.Create(ARowCount, AColCount: UInt32);
begin
  FRowCount := ARowCount;
  FColCount := AColCount;
  FTableData := TTableSpec.Create;
  Populate;
end;

destructor TTable.Destroy;
begin
  FTableData.Free;
  inherited Destroy;
end;

function TTable.IsCellEmpty(ACol, ARow: UInt32; out ACell: T): Boolean;
begin
  ACell := FTableData[KeyOf(ACol, ARow)];
  Result := IsCellEmpty(ACell);
end;

function TTable.IsCellEmpty(ACol, ARow: UInt32): Boolean;
begin
  Result := IsCellEmpty(FTableData[KeyOf(ACol, ARow)]);
end;

function TTable.AsString: string;
var
  LRow, LCol : UInt32;
begin
  Result := '';
  if FColCount = 0 then Exit;
  if FRowCount = 0 then Exit;
  for LRow := 0 to FRowCount-1 do begin
    for LCol := 0 to FColCount-1 do begin
      if IsCellEmpty(GetCell(LCol, LRow)) then begin
        Result := Result + '(x)';
      end else begin
        Result := Result + '(o)';
      end;
    end;
    Result := Result + LineEnding;
  end;
end;

procedure TTable.AppendRow;
var
  i: UInt32;
begin
  Inc(FRowCount);
  for i := 0 to FColCount - 1 do begin
    FTableData.Add(KeyOf(FColCount-1, i), Default(T));
  end;
end;

procedure TTable.AppendCol;
var
  i: UInt32;
begin
  Inc(FColCount);
  for i := 0 to FRowCount - 1 do begin
    FTableData.Add(KeyOf(i, FRowCount-1), Default(T));
  end;
end;

function TTable.KeyOf(ACol, ARow: UInt32): UInt64;
begin
  Result := (UInt64(ARow) shl 32) or UInt64(ACol);
end;

function TTable.GetCell(ACol, ARow: UInt32): T;
begin
  Result := FTableData[KeyOf(ACol, ARow)];
end;

procedure TTable.SetCell(ACol, ARow: UInt32; const AValue: T);
begin
  FTableData[KeyOf(ACol, ARow)] := AValue;
end;

function TTable.GetRowCount: UInt32;
begin
  Result := FRowCount;
end;

function TTable.GetColCount: UInt32;
begin
  Result := FColCount;
end;

function TTable.IsCellEmpty(const AValue: T): Boolean;
begin
  Result := AValue = Default(T);
end;

procedure TTable.SetColCount(AValue: UInt32);
begin
  if AValue = FColCount then Exit;
  FColCount := AValue;
end;

procedure TTable.SetRowCount(AValue: UInt32);
begin
  if AValue = FRowCount then Exit;
  FRowCount := AValue;
end;

constructor TTable.Create;
begin
  FRowCount := 0;
  FColCount := 0;
  FTableData := TTableSpec.Create;
end;

procedure TTable.InsertRow(ARow: UInt32; const AValues: array of T);
var
  i : UInt32;
begin
  if ARow > FRowCount then
    raise Exception.Create('Invalid row index for insertion.');

  if Length(AValues) <> FColCount then
    raise Exception.Create('Number of values must match the number of columns.');

  for i := 0 to FColCount-1 do begin
    FTableData.Add(KeyOf(i, ARow), AValues[i]);
  end;

  Inc(FRowCount);
end;

procedure TTable.InsertCol(ACol: UInt32; const AValues: array of T);
var
  i : UInt32;
begin
  if ACol > FRowCount then
    raise Exception.Create('Invalid col index for insertion.');

  if Length(AValues) <> FRowCount then
    raise Exception.Create('Number of values must match the number of rows.');

  for i := 0 to FRowCount-1 do begin
    FTableData.Add(KeyOf(ACol, i), AValues[i]);
  end;

  Inc(FColCount);
end;

procedure TTable.RemoveRow(ARow: UInt32);
var
  i: UInt32;
begin
  if ARow >= FRowCount then
    raise Exception.Create('Invalid row index for removal.');

  for i := 0 to FColCount-1 do begin
    FTableData.Remove(KeyOf(i, ARow));
  end;

  Dec(FRowCount);
end;

procedure TTable.RemoveCol(ACol: UInt32);
var
  i: UInt32;
begin
  if ACol >= FColCount then
    raise Exception.Create('Invalid col index for removal.');

  for i := 0 to FRowCount-1 do begin
    FTableData.Remove(KeyOf(ACol, i));
  end;

  Dec(FColCount);
end;

function TTable.CellOf(AValue: T): TCell;
var
  LCol, LRow: Integer;
begin
  Result.Col := 0;
  Result.Row := 0;
  for LCol := 0 to FColCount-1 do begin
    for LRow := 0 to FRowCount-1 do begin
      if AValue = FTableData[KeyOf(LCol, LRow)] then begin
        Result.Col := LCol;
        Result.Row := LRow;
        Break;
      end;
    end;
  end;
end;

function TTable.CellOfTarget(AValue: T; out ATargetFound: Boolean): TCell;
var
  LCol, LRow: Integer;
begin
  ATargetFound := False;
  Result.Col := 0;
  Result.Row := 0;
  for LCol := 0 to FColCount-1 do begin
    for LRow := 0 to FRowCount-1 do begin
      if AValue = FTableData[KeyOf(LCol, LRow)] then begin
        ATargetFound := True;
        Result.Col := LCol;
        Result.Row := LRow;
        Break;
      end;
    end;
  end;
end;

procedure TTable.Clear;
begin
  FTableData.Clear;
end;

procedure TTable.Empty;
var
  LCol : UInt32;
  LRow : UInt32;
begin
  for LCol := 0 to FColCount-1 do begin
    for LRow := 0 to FRowCount-1 do begin
      FTableData[KeyOf(LCol, LRow)] := Default(T);
    end;
  end;
end;

procedure TTable.Populate;
var
  LCol, LRow: UInt64;
begin
  if (FColCount = 0) or (FRowCount = 0) then Exit;

  for LCol := 0 to FColCount-1 do begin
    for LRow := 0 to FRowCount-1 do begin
      FTableData.Add(KeyOf(LCol, LRow), Default(T));
    end;
  end;
end;

procedure TTable.Invalidate;
begin
//   LColCount := FColCount;
//   if AValue > LColCount then begin
//     for i := LColCount to AValue-1 do begin
//       AppendCol;
//     end;
//   end else begin
//     for i := AValue to LColCount-1 do begin
//       RemoveCol(i);
//     end;
//   end;
//
//   LRowCount := FRowCount;
//   if AValue > LRowCount then begin
//     for i := LRowCount to AValue-1 do begin
//       AppendRow;
//     end;
//   end else begin
//     for i := AValue to LRowCount-1 do begin
//       RemoveRow(i);
//     end;
//   end;
end;

end.
