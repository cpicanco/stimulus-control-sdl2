unit eyelink.constants;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

type

  PCCDBS = ^TCCDBS;
  TCCDBS = record
    userdata: Pointer;
    buttons: array[0..255] of AnsiChar;
    internal: Pointer;
  end;

  TGetExButtonStatesFunction = function (accdbs: PCCDBS): Int32; cdecl;

const
  EXTERNAL_DEV_NONE : TGetExButtonStatesFunction = nil;
  EXTERNAL_DEV_CEDRUS : TGetExButtonStatesFunction = nil;
  EXTERNAL_DEV_SYS_KEYBOARD : TGetExButtonStatesFunction = nil;

implementation

initialization
  EXTERNAL_DEV_NONE := TGetExButtonStatesFunction(Pointer(0));
  EXTERNAL_DEV_CEDRUS := TGetExButtonStatesFunction(Pointer(1));
  EXTERNAL_DEV_SYS_KEYBOARD := TGetExButtonStatesFunction(Pointer(2));

end.

