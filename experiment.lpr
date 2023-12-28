program experiment;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
    {$IFDEF UseCThreads}
      cthreads, cmem,
    {$ENDIF}
  {$ENDIF}
  {$IFDEF HASAMIGA}athreads,{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  SysUtils,
  Forms, Forms.Main, Forms.Modal.SpeechValidation, Forms.Main.Misc;

{$R *.res}

begin
  if FileExists('heaptrc.txt') then begin
    DeleteFile('heaptrc.txt');
  end;
  {$IF Declared(heaptrc)}
  SetHeapTraceOutput('heaptrc.txt');
  {$ENDIF}
  Randomize;
  RequireDerivedFormResource:=True;
  Application.Title := 'Stimulus Control';
  Application.Scaled := True;
  Application.Initialize;
  Application.CreateForm(TFormBackground, FormBackground);
  Application.CreateForm(TFormMisc, FormMisc);
  Application.CreateForm(
    TFormManualSpeechValidation, FormManualSpeechValidation);

  Application.Run;
end.

