{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.output;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

  procedure Print(S : string);

implementation

const
  DebugOutputFilename : string = 'DebugOutput.txt';

var
  DebugOutput : TextFile;

procedure Print(S: string);
begin
  WriteLn(DebugOutput, S);
  System.Flush(DebugOutput);
end;

initialization
  AssignFile(DebugOutput, DebugOutputFilename);
  if FileExists(DebugOutputFilename) then
    System.Erase(DebugOutput);
  System.Rewrite(DebugOutput);
  System.Append(DebugOutput);

finalization
  System.Close(DebugOutput);

end.

