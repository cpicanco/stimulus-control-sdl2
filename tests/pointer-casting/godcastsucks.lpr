program godcastsucks;

uses eyelink.constants;

  function enable_external_calibration_device(
    buttonStatesfcn: TGetExButtonStatesFunction
    {other arguments removed for simplicity}): Int32; cdecl;
  var
    Statesfcn : TGetExButtonStatesFunction;
  begin
    if buttonStatesfcn = nil then
    begin
      WriteLn('Function is nil');
    end else begin
      WriteLn('Function is not nil');
    end;
    Result := Int32(buttonStatesfcn);
    case Result of
      0 : { do something };
      1 : { do something };
      2 : { do something }
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
  WriteLn(enable_external_calibration_device(TGetExButtonStatesFunction(EXTERNAL_DEV_CEDRUS)));
  WriteLn(enable_external_calibration_device(@ExButtonStatesFunction));
  ReadLn;
end.

