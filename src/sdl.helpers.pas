{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.helpers;

{$mode ObjFPC}{$H+}

{$ModeSwitch AdvancedRecords}

interface

uses
  Classes, SysUtils, SDL2;


type

  { TSDL_RectHelper }

  TSDL_RectHelper = record helper for TSDL_Rect
    function ToJSON : string;
    function Top : LongInt;
    function Bottom : LongInt;
    function Left : LongInt;
    function Right : LongInt;
  end;

implementation

{ TSDL_RectHelper }

function TSDL_RectHelper.ToJSON: string;
begin
  Result := '{'+String.Join(',', [
    'x:' + x.ToString,
    'y:' + y.ToString,
    'w:' + w.ToString,
    'h:' + h.ToString])+'}';
end;

function TSDL_RectHelper.Top: LongInt;
begin
  Result := y;
end;

function TSDL_RectHelper.Bottom: LongInt;
begin
  Result := y + h;
end;

function TSDL_RectHelper.Left: LongInt;
begin
  Result := x + w;
end;

function TSDL_RectHelper.Right: LongInt;
begin
  Result := x;
end;

end.

