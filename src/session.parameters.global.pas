unit session.parameters.global;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, sdl.app.grids.types, sdl.app.controller.types;

type

  TGlobalTrialParameters = record
    Cursor : integer;
    LimitedHold : integer;
    InterTrialInterval : integer;
    TimeOutInterval : integer;
    HasConsequence : Boolean;
    FontName : string;
    FixedSamplePosition : UInt8;
    FixedComparisonPosition : UInt8;
    ComparisonPositions : array of UInt8;
    GridOrientation : TGridOrientation;
    FontSize : integer;
    ShowModalFormForSpeechResponses : Boolean;
    AudioPromptForText : string;
    RecordingSeconds : UInt8;
    ShouldRestartAtBlockStart : Boolean;
  end;

var
  GlobalTrialParameters : TGlobalTrialParameters;

implementation


end.

