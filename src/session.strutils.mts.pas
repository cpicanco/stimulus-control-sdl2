{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.strutils.mts;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, sdl.app.audio.contract;



function GetWordValue(const AParameters: TStringList; AIsSample : boolean;
  AIndex : integer = -1) : string;

function GetFontName(const AParameters: TStringList) : string;
function GetAudioPromptForText(const AParameters: TStringList) : ISound;

function HasTextPrompt(const AParameters: TStringList) : Boolean;
function HasDAPAAPPrompt(const AParameters: TStringList) : Boolean;

function GetTotalLoopsValue(const AParameters: TStringList) : integer;

implementation

uses
  sdl.app.audio,
  session.strutils,
  session.parameters.global,
  session.constants.mts,
  session.constants.trials.dapaap;

function GetWordValue(const AParameters: TStringList; AIsSample: boolean;
  AIndex: integer): string;
begin
  if AIsSample then begin
    Result := AParameters.Values[MTSKeys.SampleKey+(AIndex+1).ToString];
  end else begin
    Result := AParameters.Values[MTSKeys.ComparisonKey+(AIndex+1).ToString];
  end;
end;

function GetFontName(const AParameters: TStringList): string;
begin
  Result := AParameters.Values[MTSKeys.FontNameKey];
  if Result.IsEmpty then begin
    Result := GlobalTrialParameters.FontName;
  end;
end;

function GetAudioPromptForText(const AParameters: TStringList): ISound;
var
  LSoundFileName : string;
begin
  LSoundFileName := AParameters.Values[MTSKeys.PromptKey];
  if LSoundFileName.IsEmpty then begin
    Result := SDLAudio.SoundFromName(
      GlobalTrialParameters.AudioPromptForText);
  end else begin
    Result := SDLAudio.LoadFromFile(AsAsset(LSoundFileName));
  end;
end;

function HasTextPrompt(const AParameters: TStringList): Boolean;
begin
  Result := StrToBooldef(AParameters.Values[MTSKeys.HasTextPromptKey], True);
end;

function HasDAPAAPPrompt(const AParameters: TStringList): Boolean;
var
  LHasPrompt : string;
begin
  Result := StrToBooldef(AParameters.Values[MTSKeys.HasPromptKey], False);
end;

function GetTotalLoopsValue(const AParameters: TStringList): integer;
begin
  Result :=
    StrToIntDef(AParameters.Values[ParserTrialsDAPAAP.TotalLoopsKey],
      GlobalTrialParameters.DefaultAudioLoops);
end;

end.

