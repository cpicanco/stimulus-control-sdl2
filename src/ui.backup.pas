{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit ui.backup;

{$mode ObjFPC}{$H+}

interface

procedure DoConfigurationsBackup;
procedure RestoreConfigurationsBackup;

implementation

uses
  Classes, SysUtils, SDL2, FileUtil, Dialogs;

const
  GINI = '.ini';
  GBKP = '.bkp';

var
  GConfigurations : string = 'configurations';
  GConfigurationsGlobal : string = 'configurations_global';

procedure Copy(ASrc, ADst : string);
begin
  if FileExists(GConfigurations+ASrc) then begin
    CopyFile(GConfigurations+ASrc, GConfigurations+ADst);
    ShowMessage(GConfigurations+ADst);
  end;

  if FileExists(GConfigurationsGlobal+ASrc) then begin
    CopyFile(GConfigurationsGlobal+ASrc, GConfigurationsGlobal+ADst);
    ShowMessage(GConfigurations+ADst);
  end;
end;

procedure DoConfigurationsBackup;
begin
  Copy(GINI, GBKP);
end;

procedure RestoreConfigurationsBackup;
begin
  Copy(GBKP, GINI);
end;

initialization
  GConfigurations := ConcatPaths([SDL_GetBasePath(), GConfigurations]);
  GConfigurationsGlobal := ConcatPaths([SDL_GetBasePath(), GConfigurationsGlobal]);

end.

