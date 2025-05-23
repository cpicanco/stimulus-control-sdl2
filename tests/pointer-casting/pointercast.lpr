program pointercast;
const C1 : integer = 1;

function Foo(P : Pointer) : integer; cdecl;
begin
  Result := Integer(P);
end;

begin
  writeln('Pointer(C1) => Integer(P) = ', Foo(Pointer(C1)));
  ReadLn;
end.

