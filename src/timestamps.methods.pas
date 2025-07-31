{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit timestamps.methods;

{$mode objfpc}{$H+}
{$MODESWITCH TYPEHELPERS}

interface

uses  SysUtils, Math;


function GetLatency(AStart, ALatency : Float) : string;
function Elapsed : Float;

type


  { TFloatHelper }

  TFloatHelper = type helper for Float
    function Elapsed: Float;
    function ToString: string;
  end;

var
  TimeStart : Float;

implementation

uses timestamps;

function GetLatency(AStart, ALatency: Float): string;
begin
  if ALatency > 0 then begin
      Result := (ALatency - AStart).ToString;
    end else begin
      Result := 'NA';
    end;
end;

function Elapsed: Float;
begin
  Result := ClockMonotonic - TimeStart;
end;

{ TFloatHelper }

function TFloatHelper.Elapsed: Float;
begin
  Result := Self - TimeStart;
end;

function TFloatHelper.ToString: string;
begin
  Result := FloatToStrF(Self, ffFixed, 0, 9);
end;


end.

