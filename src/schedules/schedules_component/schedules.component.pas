{
  Schedules
  Copyright (C) 2010-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Schedules.Component;

{$mode objfpc}{$H+}

interface

procedure Register;

implementation

uses
  Classes, Schedules;

procedure Register;
begin
  RegisterComponents('Stimulus Control',[TSchedule]);
end;

end.
