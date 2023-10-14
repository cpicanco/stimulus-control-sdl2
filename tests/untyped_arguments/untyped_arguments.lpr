program untyped_arguments;

type
  TIntegerArray = specialize TArray<integer>;

var
  A : TIntegerArray;

procedure DoIt(var arg);
var
  i : integer;
begin
  // TIntegerArray(@arg)[0]; // address is an array with lenght 1
  WriteLn();

  //if @arg  TIntegerArray then
  //  for i in TIntegerArray(arg) do
  //    WriteLn(i);
end;

begin
  A := Default(TIntegerArray);
  A := TIntegerArray.Create(0, 1, 2, 3, 4, 5);
  DoIt(A);
  ReadLn;
end.

