program standalone_experiment;
uses
  {$IFNDEF NO_LCL}Interfaces,{$ENDIF}
  sdl.app
  , eyelink.classes
  {$IFDEF NO_LCL}, sdl.app.renderer.nolcl{$ENDIF}
  ;
var
  SDLApp : TSDLApplication;
  EyeLink : TEyeLink;

begin
  EyeLink := TEyeLink.Create(nil);
  EyeLink.InitializeLibraryAndConnectToDevice;
  EyeLink.DataReceiveFile;

  //SDLApp := TSDLApplication.Create;
  //EyeLink := TEyeLink.Create(nil);
  //EyeLink.InitializeLibraryAndConnectToDevice;
  //EyeLink.HostApp := SDLApp;
  //EyeLink.DoTrackerSetup;
  //EyeLink.OpenDataFile;
  //try
  //  Monitor := SDLApp.Monitor;
  //  SDLApp.SetupEvents;
  //  SDLApp.Run;
  //finally
  //  EyeLink.ReceiveDataFile;
  //  EyeLink.Free;
  //  SDLApp.Free;
  //end;
  ReadLn;
end.

