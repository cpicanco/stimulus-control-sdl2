program sdltimertest;
uses
  Interfaces // this includes the LCL widgetset
  , Classes
  , SysUtils
  , SDL2
  , ctypes
  , timestamps.methods
  , sdl.app
  , sdl.app.events.custom
  , sdl.app.output
  , sdl.timer
  , sdl.serialtimer;

type
  TMyClass = class
    procedure NotifyEvent(Sender : TObject);
  end;

procedure TMyClass.NotifyEvent(Sender: TObject);
begin
  WriteLn('Elapsed:'+Elapsed.ToString);
  WriteLn(#9+'NotifyEvent:Sender:'+Sender.ClassName);
end;

var
  MyClass : TMyClass;
  SDLApp : TSDLApplication;
  //SerialTimer : TSerialTimer;
  Timer : TSDLTimer;

begin
  // YOU SHOULD AVOID RECURRENT CALLS TO SDL TIMER
  // AS IT MAY REQUIRE TIME COMPESATION OF ITS RESULT INTERVAL
  try
    MyClass := TMyClass.Create;
    Timer := TSDLTimer.Create;
    Timer.Interval:= 2000;
    Timer.OnTimer:=@MyClass.NotifyEvent;

    SDLApp := TSDLApplication.Create;
    Timer.Start;
    SDLApp.Run;
  finally
    Timer.Free;
    MyClass.Free;
    SDLApp.Free;
  end;
end.

