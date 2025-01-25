{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.parameters.global;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, Graphics, Math, sdl.app.grids.types;

type

  TGlobalTrialParameters = record
    Cursor : integer;
    LimitedHold : UInt32;
    InterTrialInterval : UInt32;
    TimeOutInterval : UInt32;
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
    AudioLoopInterval : UInt32;
    DefaultAudioLoops : SmallInt;
    MarkerSize : SmallInt;
    HideMouse : Boolean;
    ScreenInCentimeters : Float;
    CellsSizeInCentimenter : Float;
    UseGazeAsInput : Boolean;
    UseRemoteServer : Boolean;
    SimultaneousMTS : Boolean;
  end;

var
  GlobalTrialParameters : TGlobalTrialParameters;

implementation

initialization
  with GlobalTrialParameters do begin
    Cursor := 0;
    LimitedHold := 0;
    InterTrialInterval := 0;
    TimeOutInterval := 0;
    HasConsequence := False;
    FontName := 'Arimo-Regular';
    FixedSamplePosition := 4;
    FixedComparisonPosition := 4;
    ComparisonPositions := nil;
    GridOrientation := goCustom;
    FontSize := 12;
    ShowModalFormForSpeechResponses := False;
    AudioPromptForText := '';
    RecordingSeconds := 1;
    ShouldRestartAtBlockStart := False;
    AudioLoopInterval := 0;
    DefaultAudioLoops := 1;
    MarkerSize := 100;
    ScreenInCentimeters := 39.624;
    CellsSizeInCentimenter := 6.0;
    UseGazeAsInput := False;
    SimultaneousMTS := False;
  end;
end.

