{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

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
    SubjcID : Byte;
    SessiID : Byte;
    BlockID : Integer;
    TrialID : Integer;
    StimuID : ShortInt;
    RespoID : Integer;
    Name    : string;
    function ToString : string;
  end;

implementation

uses SysUtils;

{ TStimulusID }

function TStimulusID.ToString: string;
  function Formated(AValue : integer): string;
  begin
    Result := Format('%.2d', [AValue]);
  end;
  function FormatedStimulusID : string;
  begin
    case StimuID of
        -128..-1: Result := 'S' + Formated(Abs(StimuID));
        0..127: Result := 'C' + Formated(StimuID+1);
      end;
  end;
begin
  Result := ''.Join('-', [
    Name,
    'P'+Formated(SubjcID),
    'S'+Formated(SessiID),
    'B'+Formated(BlockID),
    'T'+Formated(TrialID),
    FormatedStimulusID,
    'R'+Formated(RespoID)]);
end;

end.

