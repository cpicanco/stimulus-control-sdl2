program pointers_string;

{$mode objfpc}{$H+}

var
  S : string = 'teste';
  PS : Pointer;

begin
  WriteLn(S);
  PS := @S;
  S := '40';
  WriteLn(string(PS^));
  ReadLn;
end.

