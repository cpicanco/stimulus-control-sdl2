{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit experiments;

{$mode objfpc}{$H+}

interface

uses common.helpers;

function MakeConfigurationFile(AFilename : string) : string;

var
  ConfigurationFilename : string = '';
  ITI : integer = 4;
  LimitedHold : integer  = 1;

implementation

uses
  session.fileutils
  , experiments.trials
  , session.configurationfile
  ;

function MakeConfigurationFile(AFilename : string): string;
begin
  Result := NewConfigurationFile;
  GlobalTrialParameters.InterTrialInterval := ITI.SecondsToMiliseconds;
  GlobalTrialParameters.LimitedHold := LimitedHold.MinutesToMiliseconds;
  GlobalTrialParameters.Cursor := 1;
  Experiments.Trials.WriteToConfigurationFile(AFilename);
  ConfigurationFile.Invalidate;
  ConfigurationFile.UpdateFile;
end;

end.
