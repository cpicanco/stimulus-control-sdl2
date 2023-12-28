program generictable;

uses Generics.Table;

type
  TTable = specialize TTable<Integer>;

var
  Table : TTable;

begin
  Table := TTable.Create(3, 3);
  Table.Cells[0, 0] := 5;
  Table.Cells[1, 1] := 5;
  Table.Cells[2, 2] := 5;

  WriteLn(Table.AsString);
  WriteLn('');

  Table.ColCount := 2;

  WriteLn(Table.AsString);
  WriteLn('');

  ReadLn;
end.

