{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.renderer.types;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

var
  DELTA_TIME : ShortInt;

implementation

const
  MONITOR_HZ = 40;

initialization
  DELTA_TIME := 1000 div MONITOR_HZ;

end.

