{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit common.helpers;

{$mode ObjFPC}{$H+}

{$modeswitch TypeHelpers}

interface

uses
  SysUtils;

type

  { TTimeConverstionHelper }

  TTimeConverstionHelper = type helper(TIntegerHelper) for integer
    function SecondsToMiliseconds : integer;
    function MinutesToMiliseconds : integer;
  end;

implementation

{ TTimeConverstionHelper }

function TTimeConverstionHelper.SecondsToMiliseconds: integer;
begin
  Result := Self*1000;
end;

function TTimeConverstionHelper.MinutesToMiliseconds: integer;
begin
  Result := Self*60000
end;

end.

