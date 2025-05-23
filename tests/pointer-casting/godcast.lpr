program godcast;

{$I constants}

type

  PCCDBS = ^TCCDBS;
  TCCDBS = record
    userdata: Pointer;
    buttons: array[0..255] of AnsiChar;
    internal: Pointer;
  end;

  TGetExButtonStatesFunction = function (accdbs: PCCDBS): Int32; cdecl;

  function enable_external_calibration_device(
    buttonStatesfcn: Pointer {other arguments removed for simplicity}): Int32; cdecl;
  var
    Statesfcn : TGetExButtonStatesFunction;
  begin
    Result := Int32(TGetExButtonStatesFunction(@buttonStatesfcn^));
    case Result of
      0 : { do nothing };
    else
      begin
        Statesfcn := TGetExButtonStatesFunction(buttonStatesfcn);
        Statesfcn(nil);
        Result := -1;
      end;
    end;
  end;

function ExButtonStatesFunction(accdbs: PCCDBS): Int32; cdecl;
begin
  WriteLn('God casts sucks');
end;

begin
  WriteLn(enable_external_calibration_device(EXTERNAL_DEV_NONE));
  WriteLn(enable_external_calibration_device(@ExButtonStatesFunction));
  ReadLn;
end.

