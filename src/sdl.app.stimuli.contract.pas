{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes, fgl, sdl.app.stimulus.contract;

type
  { IStimuli }

  IStimuli = interface
    ['{6B18F44A-7450-4871-A2BB-A109FC2ED005}']
    function AsInterface : IStimuli;
    function CustomName : string;
    procedure DoExpectedResponse;
    procedure Load(AParameters : TStringList; AParent : TObject);
    procedure Start;
    procedure Stop;
  end;

implementation

end.

