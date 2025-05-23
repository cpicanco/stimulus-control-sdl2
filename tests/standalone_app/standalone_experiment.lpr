program standalone_experiment;
uses
  Interfaces, Classes, SysUtils
  , sdl.app
  //, eyelink.classes
  , sdl.app.renderer.nolcl
  , sdl.app.system.keyboard.nolcl
  , sdl.app.task
  , timestamps.methods, sdl.app.renderer.variables
  //, sdl.app.video.writer.windows
  ;


var
  Task : TKeyboardTask;

  //EyeLink : TEyeLink;
begin
  WriteLn('Press space bar);
  //EyeLink := TEyeLink.Create(nil);
  //EyeLink.InitializeLibraryAndConnectToDevice;
  //EyeLink.DataReceiveFile;


  //EyeLink := TEyeLink.Create(nil);
  //EyeLink.InitializeLibraryAndConnectToDevice;
  //EyeLink.HostApp := SDLApp;
  //EyeLink.DoTrackerSetup;
  //EyeLink.OpenDataFile;


  Task := TKeyboardTask.Create;
  SDLApp := TSDLApplication.Create;
  try
    SDLApp.SetupVideo;
    SDLApp.Events.Keyboard.RegisterOnKeyDown(Task.OnKeyDown);
    Task.SetupMonitor(SDLApp.Monitor);
    //SDLApp.SetupEvents;
    //Sleep(50);
    //VideoWriter := TVideoWriter.Create(SDLApp.Monitor);
    //VideoWriter.StartRecording;
    SDLApp.Run;
  finally
    //EyeLink.ReceiveDataFile;
    //EyeLink.Free;
    //VideoWriter.MainThreadSynchronize;
    //VideoWriter.Stop;
    //CheckSynchronize;
    SDLApp.Free;
    Task.Free;
  end;
  ReadLn;
end.

