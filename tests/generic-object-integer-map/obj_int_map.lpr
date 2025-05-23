program obj_int_map;

uses Classes, Generics.Map;

type

  TMyMap = specialize TGenericToIntegerMap<TStringList>;

var
  StringList1 : TStringList;
  StringList2 : TStringList;
  MyMap : TMyMap;

begin

  MyMap := TMyMap.Create;
  StringList1 := TStringList.Create;
  StringList2 := TStringList.Create;

  try
    MyMap.Add(StringList1, 0);
    MyMap.Add(StringList2, 0);

    WriteLn('StringList1: ', MyMap[StringList1]);
    WriteLn('StringList2: ', MyMap[StringList2]);

    MyMap[StringList1] := MyMap[StringList1] + 1;
    MyMap[StringList2] := MyMap[StringList2] + 2;


    WriteLn('StringList1: ', MyMap[StringList1]);
    WriteLn('StringList2: ', MyMap[StringList2]);


  finally
    MyMap.Free;
    StringList1.Free;
    StringList2.Free;
  end;

  ReadLn;
end.

