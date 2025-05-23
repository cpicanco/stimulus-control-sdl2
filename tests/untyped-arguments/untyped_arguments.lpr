// studying pointer sintax for managing custom types
program untyped_arguments;

type
  TIntegerArray = specialize TArray<Integer>;
  TBooleanArray = specialize TArray<Boolean>;

var
  A : TIntegerArray;
  B : TBooleanArray;

procedure DoIt(var arg);
const
  {$IFDEF CPU32}
  	TypeInfoOffset : PtrInt = -12;
  {$ELSE}
  	{$IFDEF CPU64}
    	TypeInfoOffset : PtrInt = -24;
    {$ELSE}
      {$FATAL This architecture is not supported.}
    {$ENDIF}
  {$ENDIF}
var
  i : integer;
  PDynArray: Pointer;
  PTypeInfo: PPointer;
  PExpectedTypeInfo: PPointer;
begin
  // Get the actual array data pointer
  PDynArray := PPointer(@arg)^;

  // Test if it was assigned
  if not Assigned(PDynArray) then begin
  	WriteLn('Got: Not assigned.');
    WriteLn('--------', LineEnding);
    Exit;
  end;

  // Access the type info from the array's metadata
  // Offset for type info in Free Pascal x86: -12 bytes from the data pointer
  PTypeInfo := Pointer(PDynArray) - TypeInfoOffset;
  PExpectedTypeInfo := System.TypeInfo(TIntegerArray);

  // Compare the retrieved type info with TIntegerArray's type info
  if PTypeInfo^ <> PExpectedTypeInfo^ then begin
    WriteLn('Got: Argument is not a TIntegerArray');
    WriteLn('--------', LineEnding);
  	Exit;
  end;

  // Compare the retrieved type info with TIntegerArray's type info
  WriteLn('Got: TIntegerArray with');
  for i in TIntegerArray(arg) do begin
    WriteLn(i);
  end;
  WriteLn('--------', LineEnding);
end;

begin
  WriteLn('Expected: Not assigned.');
  DoIt(A);

  WriteLn('Expected: Argument is not a TIntegerArray.');
  B := TBooleanArray.Create(True, True, False, True, False, False);
  DoIt(B);

  WriteLn('Expected: Not assigned again from Default(TIntegerArrays).');
  A := Default(TIntegerArray);
  DoIt(A);

  WriteLn('Expected: TIntegerArray with number.');
  A := TIntegerArray.Create(0, 1, 2, 3, 4, 5);
  DoIt(A);
  ReadLn;
end.

