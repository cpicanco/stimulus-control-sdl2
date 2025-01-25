{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.types;

{$mode ObjFPC}{$H+}
{$modeSwitch advancedRecords}

interface

type

  { TStimulusID }

  TStimulusID = record
    IsSample : Boolean;
    SubjcID : Byte;
    SessiID : Byte;
    BlockID : Integer;
    TrialID : Integer;
    StimuID : ShortInt;
    RespoID : Integer;
    Name    : string;
    function ToString : string;
    function ToSpeechString : string;
  end;

implementation

uses SysUtils;

{ TStimulusID }

function Formated(AValue : integer): string;
begin
  Result := Format('%.2d', [AValue]);
end;

function FormatedID(AStimuID : integer; AIsSample: Boolean) : string;
begin
  if AIsSample then begin
    Result := 'S';
  end else begin
    Result := 'C';
  end;
  Result := Result + Formated(AStimuID+1);
end;

function FormatedSpeechName(AName : string; AIsSample: Boolean) : string;
begin
  if AIsSample then begin
    Result:= AName.Replace('Speech.Sample' + #9, '')
  end else begin
    Result:= AName.Replace('Speech.Comparison' + #9, '');
  end;
end;


function TStimulusID.ToString: string;
begin
  Result := ''.Join('-', [
    Name,
    'P'+Formated(SubjcID),
    'S'+Formated(SessiID),
    'B'+Formated(BlockID+1),
    'T'+Formated(TrialID+1),
    FormatedID(StimuID, IsSample),
    'R'+Formated(RespoID)]);
end;

function TStimulusID.ToSpeechString: string;
begin
  Result := ''.Join('-', [
    'P'+Formated(SubjcID),
    'S'+Formated(SessiID),
    'B'+Formated(BlockID+1),
    'T'+Formated(TrialID+1),
    FormatedID(StimuID, IsSample),
    'R'+Formated(RespoID),
    FormatedSpeechName(Name, IsSample)]);
end;

end.

